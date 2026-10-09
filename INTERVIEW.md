# Interview brief — F05 vehicle and person tracking

Upload this file to ChatGPT and paste the prompt at the bottom. This is the full briefing for Divyanshu's data-annotation interview. Do not invent numbers, tools, or steps that are not written here.

Candidate: Divyanshu  
Repo: https://github.com/Divyanshu0230/Data-Annotation  
Assignment: Vehicle & Person Detection and Tracking, clip `F05_dense_4k`  
Company contact on the assignment mail: Rajat Patel

## What to say in one minute

I labeled one 20-second drone shot of a city intersection. It is 1,199 frames, 3840 by 2160, 60 fps. I did not skip frames and I did not resize the images. Each box is either a vehicle or a person, and the same physical object keeps one instance id until it leaves the picture. Drawing every box by hand was not realistic, so I ran a YOLO11-L model trained on VisDrone on every frame, then linked the boxes with a tracker that first removes the drone's own motion. Short detector gaps of up to 8 frames are filled with a straight line. Longer gaps have no box. If I was not sure it was the same object, I started a new id.

## Facts that must stay exact

| Fact | Value |
|---|---|
| Clip name | F05_dense_4k |
| Frames | 1,199, named frame_00001.jpg to frame_01199.jpg |
| Image size | 3840 × 2160 |
| Frame rate | 60 fps, about 20 seconds |
| Classes | 0 = vehicle, 1 = person |
| Boxes in labels.json | 401,510 |
| Instance ids | 1 through 1786 |
| Vehicles | 1,170 tracks |
| People | 616 tracks |
| Typical boxes per frame | about 335 |
| Detector | YOLO11-L fine-tuned on VisDrone |
| Weights | models/yolo11l-visdrone.pt |
| Inference size | 1920. The jpg files were not resized. Boxes are in original 3840 × 2160 pixels. |
| Detection time | about 13 minutes |
| Linking time | under a minute |
| Total effort | a few hours, mostly rules and checking, not clicking boxes |
| Annotator field | divyanshu |
| Script | annotate.py with commands detect, track, qa, all |
| Frames folder on the machine | /Users/divyanshu/Desktop/f05_frames. Not inside the git repo. |

VisDrone class map used in code:

- 0 pedestrian → person (class_id 1)
- 2 bicycle, 3 car, 4 van, 5 truck, 6 tricycle, 7 awning-tricycle, 8 bus, 9 motor → vehicle (class_id 0)
- 1 crowd → dropped. A crowd is not one object.

Output box format: `[x, y, w, h]` in pixels, origin top-left. `sequence` is the number in the filename. Every frame from 1 to 1199 is present once. An instance id never appears twice in the same frame. Ids are shared by vehicles and people, so there is never a vehicle 3 and a person 3. Ids are given in order of first appearance.

## How the system works, in the order to explain it

1. `python3 annotate.py detect` reads every frame and runs YOLO11-L at imgsz 1920. It keeps boxes with confidence at least 0.15, runs NMS, and removes people who are really part of a vehicle. It also estimates a homography from the previous frame to the current frame using optical flow, ignoring points that fall inside detections so moving cars do not get counted as camera motion. Results go to detections.npz.
2. `python3 annotate.py track` does not run the network again. It links boxes into tracks.
3. Before matching, each live box is warped by the homography, then a small velocity is added. That velocity is the object's own motion after the camera motion is removed. Near frame 1020 the drone moves about 40 pixels in one frame. Matching in the raw image would look like every object jumped.
4. Matching is Hungarian, same class only (vehicle with vehicle, person with person). High-score boxes (confidence at least 0.34) match first. Leftover tracks may match weaker boxes. A weak box can extend a track. It cannot start a new id.
5. A new id needs a higher score. Roughly: person 0.27 inside the image or 0.22 at the border, two-wheeler 0.30 or 0.25 at the border, other vehicles 0.26 or 0.22 at the border.
6. If a track is missed but still mostly inside the image, the id stays alive for up to 48 frames (0.8 seconds) and no box is written. If it is at the border, the limit is 8 frames. If less than about 28% of the predicted box is still inside the image, the id is retired. A later re-entry is a new id. The last visible boxes are truncated and cover only the visible part.
7. After linking, a track is split if the camera-compensated jump is too large to be the same object. A wrong merge is worse than a split. Very short flashes are dropped. A track is kept if it has at least 5 observations, or fewer observations with high confidence, or it touched the border while entering or leaving.
8. Box center and size are median-smoothed only inside a run of consecutive frames. A time gap is never smoothed across, because that would place the box between two positions.
9. Gaps of 1 to 8 frames are filled with a straight line. Longer gaps stay empty. If a filled box would be about 65% covered by a nearer object, that frame is left empty too.
10. Occlusion: 0 if overlap with a nearer box is under about 18%, 1 up to about half, 2 above about half. Nearer means the other box's bottom edge is lower in the image, because the view is oblique. Partly hidden boxes were not expanded to a guessed full outline. The box stays on what the detector drew.
11. `python3 annotate.py qa` draws sample frames. The real check was opening random frames, including consecutive frames, and following single cars and people. One white car kept one id until it left the edge.

## Rules and the call that was made

- Riders: the two-wheeler is always a vehicle. The rider is also a person only when the body clearly sticks up and reads as its own shape. If they are one blob, only the vehicle is kept.
- People inside a car, van, truck, or bus are not boxed.
- Each person in a group gets their own box. Never one box around a crowd.
- Auto-rickshaws are vehicles. So are bicycles, scooters, motorcycles, cars, vans, trucks, buses, and cement mixers.
- A car on a sign or billboard is not a vehicle. The model was not seen boxing those.
- Parked cars and people standing still keep one id the whole time they are in frame.
- Small and far objects are kept when the detector holds them for several frames. A box that appears for only one or two frames is dropped.
- Some dark poles, shadows, and rooftop clutter can look like a person or a bike. Unstable ones were dropped. A few that stayed steady were kept, because missing a real pedestrian is also wrong. Divyanshu is less sure about a handful of those.
- Cement mixers and the tanker have looser boxes than the cars. In the packed queue two boxes sometimes touch. Those were left as the detector drew them.
- Occlusion was not hand-estimated for every hidden pixel. It comes from overlap.

## How it was graded

| Criterion | Weight |
|---|---|
| Identity stays consistent, and ids are retired when the object leaves | 35% |
| Box quality on frames that were not hand-picked, including any interpolated frame | 30% |
| Recall of small, distant, and partly hidden objects | 20% |
| File validity: parses, all 1,199 frames once, no duplicate id in a frame | 10% |
| The note on ambiguous cases | 5% |

The file checks passed. The quality items are the ones an interviewer will push on. Do not claim every box is perfect.

## What not to say

- Do not say the frames were resized or that the model ran at full 3840. Inference size is 1920. Files stayed 3840 × 2160.
- Do not say every box was drawn in CVAT. CVAT was the tool the assignment suggested. This pass used the script.
- Do not say partly hidden boxes were extended to the full guessed shape. They were not.
- Do not say a person and a vehicle can share an instance id. They cannot.
- Do not say an id is reused when someone walks back in. It is not.
- Do not say interpolation fills the whole clip. It only fills gaps of 8 frames or fewer.
- Do not say the homography includes the traffic. Points inside detections were masked out.
- Do not invent a second model, a manual QA count, or a mAP number. None of those were measured.

## If they ask how you would do it better

Say this, in this order:

1. Sample more random frames and fix drifted or loose boxes by hand, especially trucks, tankers, and the dense queue.
2. Run the detector at a larger inference size, or on tiles, to catch smaller distant bikes and people. The full pass at 1920 was chosen because it finished in about 13 minutes.
3. For riders, look at a crop and decide person-versus-vehicle by eye on the unclear ones.
4. Keep the rule that a doubtful id match becomes a new id.

## Paste this into ChatGPT after uploading the file

You are coaching me, Divyanshu, for a data-annotation interview about the F05 drone clip. Use only INTERVIEW.md. Do not add tools, numbers, or steps that are not in the file.

First, ask me to give the one-minute explanation with no notes. Then score it out of 10 and tell me the one sentence I missed.

Then ask me 12 questions, one at a time, in simple words. Wait for my answer before the next question. After each answer, say what was right, what was wrong, and the short sentence I should have said. Cover: why all 1,199 frames, what instance id means, when an id dies, occlusion, riders, camera motion, why inference is 1920, what interpolation is allowed to do, what is still weak, and how I would improve it.

End with a 3-minute mock: you are the interviewer, I am the candidate. Interrupt me if I claim something the file says not to say. Close with the five lines I must remember.
