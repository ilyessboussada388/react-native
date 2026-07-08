import React, { useEffect, useRef, useState } from 'react';
import { Button, SafeAreaView, ScrollView, StyleSheet, TextInput, View } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';

type Detection = {
  id?: number;
  class_name?: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
};

type Metrics = {
  count: number;
  visible: number;
  fps: number;
  model?: string;
  device?: string;
  infer_ms?: number;
  process_ms?: number;
  detections?: Detection[];
};

type UploadResponse = {
  ok: boolean;
  image?: string;
  error?: string;
  metrics?: Metrics;
};

const DEFAULT_SERVER_URL = 'https://truck-package-counter.onrender.com';
const CAPTURE_INTERVAL_MS = 650;
const REQUEST_TIMEOUT_MS = 45000;
const UPLOAD_WIDTH = 640;

async function fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

export default function CameraPage() {
  const cameraRef = useRef<CameraView | null>(null);
  const sendingRef = useRef(false);
  const [permission, requestPermission] = useCameraPermissions();
  const [serverUrl, setServerUrl] = useState(DEFAULT_SERVER_URL);
  const [active, setActive] = useState(false);
  const [metrics, setMetrics] = useState<Metrics>({ count: 0, visible: 0, fps: 0, detections: [] });
  const [status, setStatus] = useState('Start the backend, then press Start analysis.');

  const cleanServerUrl = serverUrl.replace(/\/+$/, '');

  const uploadFrame = async () => {
    if (sendingRef.current || !active) return;
    if (!cameraRef.current) {
      setStatus('Camera is starting. Keep the app open for a moment...');
      return;
    }
    if (!cleanServerUrl.startsWith('http')) {
      setStatus('Server URL must start with http:// or https://');
      return;
    }

    sendingRef.current = true;
    try {
      const photo = await cameraRef.current.takePictureAsync({
        base64: false,
        quality: 0.25,
        skipProcessing: true,
        exif: false,
      });
      if (!photo?.uri) throw new Error('Camera did not return a frame.');

      const resized = await manipulateAsync(
        photo.uri,
        [{ resize: { width: UPLOAD_WIDTH } }],
        { base64: true, compress: 0.28, format: SaveFormat.JPEG }
      );
      if (!resized.base64) throw new Error('Could not prepare frame for upload.');

      const response = await fetchWithTimeout(`${cleanServerUrl}/upload-frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: resized.base64, return_image: false }),
      });
      if (!response.ok) throw new Error(`Server returned HTTP ${response.status}`);
      const data = (await response.json()) as UploadResponse;
      if (!data.ok || !data.metrics) throw new Error(data.error || 'Backend returned no metrics.');

      setMetrics(data.metrics);
      setStatus(`Connected | inference ${data.metrics.infer_ms ?? '?'} ms | total ${data.metrics.process_ms ?? '?'} ms`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus(`Analysis error: ${message.slice(0, 180)}`);
    } finally {
      sendingRef.current = false;
    }
  };

  useEffect(() => {
    if (!active) return;
    const timer = setInterval(uploadFrame, CAPTURE_INTERVAL_MS);
    uploadFrame();
    return () => clearInterval(timer);
  }, [active, cleanServerUrl]);

  const resetCount = async () => {
    setMetrics((current) => ({ ...current, count: 0, detections: [] }));
    try {
      await fetchWithTimeout(`${cleanServerUrl}/reset`, { method: 'POST' }, 15000);
      setStatus('Counter reset.');
    } catch {
      setStatus('Could not reset server counter.');
    }
  };

  const testServer = async () => {
    if (!cleanServerUrl.startsWith('http')) {
      setStatus('Server URL must start with http:// or https://');
      return false;
    }
    setStatus('Testing server connection...');
    try {
      const response = await fetchWithTimeout(`${cleanServerUrl}/health`, {}, 60000);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (!data.ok) throw new Error('Server health check did not return ok=true.');
      setStatus(`Server online. Model: ${data.model ?? 'unknown'} | Device: ${data.device ?? 'unknown'}`);
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus(`Server test failed: ${message.slice(0, 180)}`);
      return false;
    }
  };

  const toggleAnalysis = async () => {
    if (active) {
      setActive(false);
      setStatus('Analysis stopped.');
      return;
    }
    setStatus('Starting analysis. Waking server if needed...');
    const ok = await testServer();
    if (!ok) return;
    setActive(true);
    setStatus('Server online. Starting camera analysis...');
  };

  if (!permission?.granted) {
    return (
      <ThemedView style={styles.container}>
        <SafeAreaView style={styles.safeArea}>
          <View style={styles.centerContent}>
            <ThemedText type="title">Camera permission needed</ThemedText>
            <ThemedText type="small">Allow camera access so the app can analyze boxes.</ThemedText>
            <Button title="Allow camera" onPress={requestPermission} />
          </View>
        </SafeAreaView>
      </ThemedView>
    );
  }

  return (
    <ThemedView style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView contentContainerStyle={styles.scrollContent} keyboardShouldPersistTaps="handled">
          <ThemedText type="title">Phone Camera Counter</ThemedText>

          <View style={styles.metricsRow}>
            <View style={styles.metricBox}>
              <ThemedText type="smallBold" style={styles.metricLabel}>Boxes</ThemedText>
              <ThemedText type="subtitle" style={styles.metricValue}>{metrics.count ?? 0}</ThemedText>
            </View>
            <View style={styles.metricBox}>
              <ThemedText type="smallBold" style={styles.metricLabel}>Detected</ThemedText>
              <ThemedText type="subtitle" style={styles.metricValue}>{metrics.visible ?? 0}</ThemedText>
            </View>
            <View style={styles.metricBox}>
              <ThemedText type="smallBold" style={styles.metricLabel}>FPS</ThemedText>
              <ThemedText type="subtitle" style={styles.metricValue}>{Number(metrics.fps ?? 0).toFixed(1)}</ThemedText>
            </View>
          </View>

          <View style={styles.cameraPanel}>
            <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" animateShutter={false} />
            <View pointerEvents="none" style={styles.countLine} />
            {(metrics.detections ?? []).map((detection, index) => (
              <View
                key={`${detection.id ?? index}-${detection.x}-${detection.y}`}
                pointerEvents="none"
                style={[
                  styles.box,
                  {
                    left: `${detection.x * 100}%`,
                    top: `${detection.y * 100}%`,
                    width: `${detection.width * 100}%`,
                    height: `${detection.height * 100}%`,
                  },
                ]}
              >
                <ThemedText type="smallBold" style={styles.boxLabel}>
                  {Math.round((detection.confidence ?? 0) * 100)}%
                </ThemedText>
              </View>
            ))}
          </View>

          <View style={styles.controls}>
            <ThemedText type="smallBold">Backend server URL</ThemedText>
            <TextInput
              value={serverUrl}
              onChangeText={setServerUrl}
              placeholder="https://your-server.onrender.com"
              style={styles.input}
              autoCapitalize="none"
              keyboardType="url"
            />
            <View style={styles.buttonRow}>
              <View style={styles.buttonWrap}>
                <Button title={active ? 'Stop analysis' : 'Start analysis'} onPress={toggleAnalysis} />
              </View>
              <View style={styles.buttonWrap}>
                <Button title="Reset count" onPress={resetCount} />
              </View>
            </View>
            <Button title="Test server" onPress={testServer} />
            <ThemedText type="small" style={styles.statusText}>{status}</ThemedText>
            {metrics.model && <ThemedText type="small">Model: {metrics.model} | Device: {metrics.device ?? 'unknown'}</ThemedText>}
          </View>
        </ScrollView>
      </SafeAreaView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  safeArea: { flex: 1 },
  scrollContent: { padding: 16, gap: 12, paddingBottom: 36 },
  centerContent: { flex: 1, gap: 12, alignItems: 'center', justifyContent: 'center', padding: 24 },
  metricsRow: { flexDirection: 'row', gap: 8 },
  metricBox: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#d6d6d6',
    borderRadius: 8,
    padding: 10,
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  metricLabel: { color: '#111827', textAlign: 'center' },
  metricValue: { color: '#111827', textAlign: 'center' },
  cameraPanel: {
    width: '100%',
    aspectRatio: 3 / 4,
    backgroundColor: '#000',
    borderRadius: 8,
    overflow: 'hidden',
  },
  countLine: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: '65%',
    height: 3,
    backgroundColor: '#20d66b',
  },
  box: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#20d66b',
    backgroundColor: 'rgba(32, 214, 107, 0.10)',
  },
  boxLabel: {
    position: 'absolute',
    left: 0,
    top: 0,
    color: '#111',
    backgroundColor: '#20d66b',
    paddingHorizontal: 4,
    paddingVertical: 2,
  },
  controls: { gap: 8 },
  input: { borderWidth: 1, borderColor: '#d6d6d6', padding: 10, borderRadius: 8, backgroundColor: '#fff' },
  buttonRow: { flexDirection: 'row', gap: 10 },
  buttonWrap: { flex: 1 },
  statusText: { color: '#fff' },
});
