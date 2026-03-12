# Language Tutor Mobile

Expo-based React Native client for the existing `backend/` API.

## Run

```bash
cd mobile
npm install
npm start
```

## Backend URL

Set `EXPO_PUBLIC_API_BASE_URL` before starting Expo if `localhost` is not correct for your simulator or device.

Examples:

```bash
EXPO_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api npm start
EXPO_PUBLIC_API_BASE_URL=http://192.168.1.10:8000/api npm start
```

## Notes

- The backend remains unchanged for now.
- Mobile recording currently records audio locally and uploads the finished file for analysis.
- Browser-style live streaming transcription is not reproduced yet because the web implementation depends on `MediaRecorder` plus websocket chunk streaming.
