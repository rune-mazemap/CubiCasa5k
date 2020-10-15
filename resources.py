import pickle
import logging

from aioify import aioify
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

import numpy as np
import torch
import torch.nn.functional as F
from floortrans.loaders import RotateNTurns


logger = logging.getLogger()
logger.setLevel(logging.INFO)

rot = RotateNTurns()


async def check_health(request: Request) -> Response:
    return JSONResponse({'status': 'healthy'})

def run_inference(model, sample):
    image = torch.stack((2. * (torch.tensor(sample['image'].astype(np.float32)) / 255.) - 1.,), 0).cuda()
    width = image.shape[3]
    height = image.shape[2]
    
    n_classes = 30
    split = [21, 7, 2]
    with torch.no_grad():
        img_size = (height, width)

        rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
        pred_count = len(rotations)
        predictions = torch.zeros([pred_count, n_classes, height, width])
        for i, r in enumerate(rotations):
            forward, back = r
            # We rotate first the image
            rot_image = rot(image, 'tensor', forward)
            pred = model(rot_image)
            # We rotate prediction back
            pred = rot(pred, 'tensor', back)
            # We fix heatmaps
            pred = rot(pred, 'points', back)
            # We make sure the size is correct
            pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
            # We add the prediction to output
            predictions[i] = pred[0]

    raw_prediction = torch.mean(predictions, 0, True)

    prediction = {'track_id': sample['track_id']}
    rooms_pred = F.softmax(raw_prediction[0, 21:21+7], 0).cpu().data.numpy()
    prediction['rooms'] = np.argmax(rooms_pred, axis=0)

    icons_pred = F.softmax(raw_prediction[0, 21+7:], 0).cpu().data.numpy()
    prediction['icons'] = np.argmax(icons_pred, axis=0)
    # TODO junctions
    return prediction


aio_run_inference = aioify(obj=run_inference)


async def segment(request: Request) -> Response:
    content_type = request.headers['content-type']
    request_payload = await request.body()
    sample = pickle.loads(request_payload)
    prediction = await aio_run_inference(model=request.app.state.MODEL, sample=sample)
    response_payload = pickle.dumps(prediction)
    return Response(response_payload, media_type='application/octet-stream')