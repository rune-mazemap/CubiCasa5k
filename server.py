import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import uvicorn

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.routing import Route

from resources import check_health, segment
from floortrans.models import get_model


def setup_model(checkpoint_path):
    if checkpoint_path:
        model = get_model('hg_furukawa_original', 51)

        n_classes = 30
        split = [21, 7, 2]
        model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
        model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        model.cuda()
        
        return model


async def http_exception(_request, exc):
    return JSONResponse({'detail': exc.detail}, status_code=exc.status_code)


exception_handlers = {HTTPException: http_exception}
routes = [
    Route('/health', check_health, methods=['GET'], name='check_health'),
    Route('/segment', segment, methods=['POST'], name='segment')
]
app = Starlette(debug=True, routes=routes, exception_handlers=exception_handlers)
app.state.MODEL = setup_model(os.environ.get('CHECKPOINT_PATH'))


def main():
    parser = ArgumentParser(description='Segmentation service')
    parser.add_argument('-c', '--checkpoint', help='Path to checkpoint file to use', required=True)
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        parser.error(f'{args.checkponit} does not exist')
    
    return str(checkpoint_path)
    
if __name__ == '__main__':
    os.environ['CHECKPOINT_PATH'] = main()
    uvicorn.run('server:app', host='0.0.0.0', port=9876, log_level='info')