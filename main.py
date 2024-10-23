
import cv2
from program.binary_melanoma_detector import BinaryMelanomaDetector
import security_check
 

if __name__ == '__main__':
    
    img_path_name = "C:/Users/Agustin/portfolio/medical_imaging_projects/s3_skin_lesion_detection_ml/code/binary/program/images/original/train/benign/nev00010.jpg"
    # img_path_name = "C:/Users/Agustin/portfolio/medical_imaging_projects/s3_skin_lesion_detection_ml/code/binary/program/images/original/train/malign/mal00002.jpg"
    img = cv2.imread(img_path_name)

    security_message, is_secure = security_check.check_image(img)
    if not is_secure:
        raise Exception('Security check failed: {}'.format(security_message))
     
    pred, prob, img_result = BinaryMelanomaDetector().test(img)
    print(f"Prediction: {pred}, Probability: {prob}")

    cv2.imshow("img_result", img_result)
    cv2.waitKey(0)
