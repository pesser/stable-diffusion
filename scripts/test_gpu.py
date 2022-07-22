import socket
try:
    import torch
    n_gpus = torch.cuda.device_count()
    print(f"checking {n_gpus} gpus.")
    for i_gpu in range(n_gpus):
        print(i_gpu)
        device = torch.device(f"cuda:{i_gpu}")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        data = torch.randn([4, 640, 32, 32], dtype=torch.float, device=device, requires_grad=True)
        net = torch.nn.Conv2d(640, 640, kernel_size=[3, 3], padding=[1, 1], stride=[1, 1], dilation=[1, 1], groups=1)
        net = net.to(device=device).float()
        out = net(data)
        out.backward(torch.randn_like(out))
        torch.cuda.synchronize()
except RuntimeError as err:
    import requests
    import datetime
    import os
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    hostname = socket.gethostname()
    ts = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    resp = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
    print(f'ERROR at {ts} on {hostname}/{resp.text} (CUDA_VISIBLE_DEVICES={device}): {type(err).__name__}: {err}', flush=True)
    raise err
else:
    print(f"checked {socket.gethostname()}")
