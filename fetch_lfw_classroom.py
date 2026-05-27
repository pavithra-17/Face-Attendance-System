import os
import cv2
from sklearn.datasets import fetch_lfw_people
from app import app, db, User

def download_and_populate_lfw():
    print("⏳ Connecting to the server to fetch the LFW dataset... Please hold on!")
    
    # Download the dataset automatically via scikit-learn
    # We ask for color images and look for people who have at least 20 images available
    lfw_dataset = fetch_lfw_people(min_faces_per_person=20, color=True, resize=None)
    
    images = lfw_dataset.images  # The image pixel matrices
    target_ids = lfw_dataset.target  # The numerical ID mappings
    names_registry = lfw_dataset.target_names  # The actual human string names
    
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    
    # We want a clean pool of exactly 60 unique students
    max_students = 60
    unique_tracked_ids = set()
    
    print(f"📦 Successfully loaded dataset. Processing {max_students} target profiles...")
    
    with app.app_context():
        added_profiles_count = 0
        
        # Loop through the raw dataset arrays
        for idx, img_matrix in enumerate(images):
            person_id = target_ids[idx]
            raw_name = names_registry[person_id] # Example: "George_W_Bush" or "Colin_Powell"
            
            # If we've already saved a picture for this specific individual, skip to keep it unique
            if person_id in unique_tracked_ids:
                continue
                
            # Formatting: Change "George_W_Bush" to clean layout words "George Bush"
            clean_name = raw_name.replace('_', ' ').title()
            file_safe_name = raw_name.lower()
            
            # Save the image matrix to your local folder as a real JPG image file
            # Convert RGB array formatting to OpenCV's required BGR array layout structure
            bgr_frame = cv2.cvtColor((img_matrix * 255).astype('uint8'), cv2.COLOR_RGB2BGR)
            file_path = os.path.join(output_dir, f"{file_safe_name}_front.jpg")
            cv2.imwrite(file_path, bgr_frame)
            
            # Generate a clean, fake academic institutional email address
            mock_email = f"{file_safe_name}@student.university.edu"
            
            # Push account profile properties to the local SQLite database register if absent
            existing_user = User.query.filter_by(email=mock_email).first()
            if not existing_user:
                new_student = User(name=clean_name, email=mock_email, role="student")
                db.session.add(new_student)
                
            unique_tracked_ids.add(person_id)
            added_profiles_count += 1
            
            if added_profiles_count >= max_students:
                break
                
        db.session.commit()
        
        # Clear out any stale deepface pickle data caching layers to prevent conflicts
        for file in os.listdir(output_dir):
            if file.endswith('.pkl'):
                os.remove(os.path.join(output_dir, file))
                
        print(f"🎉 COMPLETE: Automatically generated and registered {added_profiles_count} unique LFW profiles into your workspace environment!")

if __name__ == "__main__":
    download_and_populate_lfw()