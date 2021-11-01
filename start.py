import cv2
import os
import time
import torch
import argparse
import numpy as np
import torchvision.transforms as T

from PIL import Image
from model import siamese_model
from facenet_pytorch import MTCNN, InceptionResnetV1

torch.cuda.empty_cache()


def main():
    cooldown_limit = 0.5  # Minimum time needed for model to confirm change in number of people in frame
    regular_check_limit = 3  # Regular classification check
    db_path = os.path.join(os.path.dirname(__file__), "database/")
    load_from_file = True
    yolov5_type = "yolov5m"
    screen_size = (800, 600)
    scale = (1, 1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-db", "--db_path", help="Use absolute path . Default path : " + db_path
    )
    parser.add_argument(
        "-load",
        "--load_from_file",
        help="[TRUE] if you want to load reference embeddings from previously generated file , [FALSE] if you want to recompile or create new embeddings for the reference images . Default is set to TRUE",
    )
    parser.add_argument(
        "-yolov5",
        "--yolov5_type",
        help="Enter which yolov5 model you want to use : [yolov5s] ,[yolov5m] ,[yolov5l] ,[yolov5x] . Default type : yolov5m ",
    )
    parser.add_argument(
        "-cdl",
        "--cooldown_limit",
        help=f" Lower the cooldown higher the precision higher the memory usage . Default value : {cooldown_limit}s",
    )
    parser.add_argument(
        "-rcl",
        "--regular_check_limit",
        help=f" Helps in correcting previous errors by either the camera or the program . Default value : {regular_check_limit}s",
    )
    parser.add_argument(
        "-size",
        "--screen_size",
        help=f"Set Default screen size for the webcam feed : [(SCREEN_W,SCREEN_H)] . Default size : {screen_size} ",
    )
    parser.add_argument(
        "-scale",
        "--scale",
        help=f"Set Default scale for the webcam feed : [(SCALE_X,SCALE_Y)] . Default size : {scale} ",
    )

    args = parser.parse_args()
    if args.db_path:
        db_path = args.db_path

    if args.load_from_file:
        if args.load_from_file.upper() == "FALSE":
            load_from_file = False

    if args.yolov5_type:
        yolov5_type = args.yolov5_type

    if args.cooldown_limit:
        cooldown_limit = args.cooldown_limit

    if args.regular_check_limit:
        regular_check_limit = args.regular_check_limit

    if args.screen_size:
        screen_size = ()
        new_screen_size = args.screen_size[1:-1].rstrip()
        new_screen_size = new_screen_size.split(",")
        for i in new_screen_size:
            screen_size.append(int(i))

    if args.scale:
        scale = ()
        new_scale = args.scale[1:-1].rstrip()
        new_scale = new_scale.split(",")
        for i in new_scale:
            scale.append(int(i))

    # Initializing all the models and reference images
    device, classes, loader, reference_cropped_img, yolov5, resnet, mtcnn, model = init(
        load_from_file=load_from_file, db_path=db_path, yolov5_type=yolov5_type
    )

    # Initializing cooldown clocks and Face-Recognition paramaters
    n_people = 0  # Number of people confirmed after cooldown
    cooldown = 0
    new_frame_time = 0
    prev_frame_time = 0
    regular_check_cooldown = 0

    classify_faces = True  # Wheather to use MTCNN to classify faces
    start_cooldown = False  # Starts when there is change in number of people
    person_names = []

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        new_frame_time = time.time()
        time_diff = new_frame_time - prev_frame_time
        fps = int(1 / (time_diff))
        regular_check_cooldown = regular_check_cooldown + time_diff
        fps = cap.get(cv2.CAP_PROP_FPS)

        boxes_info = yolov5(frame).xyxy[0].cpu().numpy().tolist()
        person_boxes = []  # selecting person class alone

        ith_n_people = 0  # number of person in frame at the moment (It might be wrong and is confirmed through cooldown)
        for i in boxes_info:
            if i[5] == 0:  # class for person is 0
                person_boxes.append(tuple(i[:4]))
                ith_n_people = ith_n_people + 1

        if ith_n_people != n_people and start_cooldown == False:
            start_cooldown = True

        if ith_n_people == n_people:
            cooldown = 0
            start_cooldown = False

        if regular_check_cooldown >= regular_check_limit:
            regular_check_cooldown = 0
            classify_faces = True

        if start_cooldown:
            cooldown = cooldown + time_diff

        if (
            cooldown >= cooldown_limit
        ):  # Confirming if the change in number is slight error
            n_people = ith_n_people
            cooldown = 0
            start_cooldown = False
            classify_faces = True  # Number of people in frame is changed so we feed the frame into MTCNN

        if (
            classify_faces and n_people == 0
        ):  # If number of peopel is 0 in frame then there is no need to classify
            person_names = []
            face_boxes = []
            face_name = []
            classify_faces = False

        if classify_faces:
            # Initializing new boxes and person name
            person_names = []
            face_boxes = []
            face_name = []

            classify_faces = False
            boxes, probs, points = mtcnn.detect(frame[:, :, ::-1], landmarks=True)

            if boxes is not None:
                for box in boxes:  # classifying predicted boxes
                    predicted_class, similarity = classify(
                        box,
                        frame,
                        loader,
                        resnet,
                        model,
                        reference_cropped_img,
                        classes,
                        device,
                    )
                    face_boxes.append(box)
                    if predicted_class == -1:
                        face_name.append("Stranger")
                    else:
                        face_name.append(predicted_class)

            for i in person_boxes:
                temp_name = ""
                new_max = 0
                for j, k in zip(face_boxes, face_name):
                    iou = IOU(
                        i, j, screen_size=tuple(frame.shape[:2])
                    )  # The box for the person and box for the person's face must intersect the highest
                    if new_max < iou:
                        new_max = iou
                        temp_name = k

                person_names.append(temp_name)

        check_repeat = (
            []
        )  # Make faces of same person cannot present at the same time (prevent error due to yolov5 smaller models)
        for i, j in zip(person_boxes, person_names):
            if j in check_repeat:
                continue
            check_repeat.append(j)
            x_min, y_min, x_max, y_max = i
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

            # color coding boxes

            color = (0, 255, 0)  # Green
            if j == "Stranger":
                check_repeat.pop()
                color = (0, 0, 255)  # Red

            frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (color), 2)
            cv2.putText(
                frame,
                f"{j}",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        # changing frame size

        frame = cv2.resize(
            frame, screen_size, fx=scale[0], fy=scale[1], interpolation=cv2.INTER_AREA
        )

        prev_frame_time = new_frame_time
        cv2.putText(
            frame, f"FPS:{fps}", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
        )
        cv2.imshow("WebCam", frame)

        c = cv2.waitKey(1)  # User input
        if c == 27:  # ASCII for ESC
            break

    cap.release()
    cv2.destroyAllWindows()


def classify(box, frame, loader, resnet, model, reference_cropped_img, classes, device):

    input_img = frame[:, :, ::-1]  # converting BGR ---> RGB
    box = (np.array(box)).astype(int)
    input_img = np.array(input_img)[box[1] : box[3] + 1, box[0] : box[2] + 1].copy()
    input_img = cv2.resize(input_img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    input_img = loader((input_img - 127.5) / 128.0).type(
        torch.FloatTensor
    )  # Normalizing and converting to tensor

    THRESHOLD = 0.4  # Minimum similairty required to be classified among classes

    similarity = []
    target_embeddings = resnet(input_img.unsqueeze(0).to(device)).reshape((1, 1, 512))

    for j in reference_cropped_img:
        j_embeddings = resnet(j.unsqueeze(0).to(device)).reshape((1, 1, 512))
        similarity.append(
            model(target_embeddings, j_embeddings).item()
        )  # feeding embeddings into siamese model

    max_similarity = max(similarity)
    if max_similarity >= THRESHOLD:
        predicted_class = classes[similarity.index(max_similarity)]
        return predicted_class, max_similarity

    return -1, -1


def IOU(box1, box2, screen_size=(480, 640)):  # calculating IOU
    boolean_box1 = np.zeros(screen_size, dtype=bool)
    boolean_box2 = np.zeros(screen_size, dtype=bool)

    x_min, y_min, x_max, y_max = box1
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            boolean_box1[y][x] = True

    x_min, y_min, x_max, y_max = box2
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            boolean_box2[y][x] = True

    overlap = boolean_box1 * boolean_box2  # Logical AND
    union = boolean_box1 + boolean_box2  # Logical OR

    return overlap.sum() / float(union.sum())


def init(load_from_file=False, db_path=None, yolov5_type="yolov5m"):
    margin = 0
    dirname = os.path.dirname(__file__)
    if db_path is not None:
        db_path = os.path.join(dirname, "database/")

    model_path = os.path.join(dirname, "saved_models/siamese_model")
    database_embeddings_path = os.path.join(db_path, "database_embeddings")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classes = []
    reference_img = []
    reference_cropped_img = []

    # Loading weights
    model = siamese_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    # Initializing models
    yolov5 = torch.hub.load("ultralytics/yolov5", yolov5_type)
    mtcnn = MTCNN(image_size=128, margin=margin).eval()
    resnet = InceptionResnetV1(pretrained="vggface2").to(device).eval()

    loader = T.Compose([T.ToTensor()])

    if load_from_file == True:
        if os.path.exists(database_embeddings_path):
            reference_cropped_img = torch.load(database_embeddings_path)["reference"]

        else:
            load_from_file = False

    if load_from_file == False:

        for i in os.listdir(db_path):
            classes.append(i)

        for i in classes:
            reference_img.append(
                Image.open(db_path + i + "/" + os.listdir(db_path + i)[0])
            )

        for i in range(len(reference_img)):
            boxes, probs, points = mtcnn.detect(reference_img[i], landmarks=True)

            boxes = (np.array(boxes[0])).astype(int)
            input_img = np.array(reference_img[i])[
                boxes[1] : boxes[3] + 1, boxes[0] : boxes[2] + 1
            ].copy()
            input_img = cv2.resize(
                input_img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC
            )
            input_img = loader((input_img - 127.5) / 128.0).type(torch.FloatTensor)
            reference_cropped_img.append(input_img)

        torch.save({"reference": reference_cropped_img}, database_embeddings_path)

    return device, classes, loader, reference_cropped_img, yolov5, resnet, mtcnn, model


if __name__ == "__main__":
    main()
