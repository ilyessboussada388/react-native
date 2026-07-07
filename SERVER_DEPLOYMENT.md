# Deploy The Box Counter Backend

The phone app can use a cloud server instead of your PC. The backend is ready for Render using `render.yaml` and `backend/Dockerfile`.

## 1. Push This Project To GitHub

Make sure these files are included:

- `backend/camera_counter_service.py`
- `backend/models/package_label_best.pt`
- `backend/settings.json`
- `backend/Dockerfile`
- `render.yaml`

## 2. Create The Render Service

1. Go to Render.
2. Create a new **Blueprint**.
3. Connect the GitHub repo.
4. Render will read `render.yaml`.
5. Deploy the `truck-package-counter` service.

The server URL will look like:

```text
https://truck-package-counter.onrender.com
```

Open this in a browser to test:

```text
https://truck-package-counter.onrender.com/health
```

It should return JSON with `"ok": true`.

## 3. Use The URL In The App

In the phone app, set **Backend server URL** to your Render URL:

```text
https://truck-package-counter.onrender.com
```

Then press **Start analysis**.

## Notes

Render free servers sleep when unused. The first request can be slow. For better FPS, use a paid CPU server or GPU server.
