import React, { useState } from 'react';
import { SafeAreaView, View, Button, StyleSheet, TextInput } from 'react-native';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { WebView } from 'react-native-webview';

export default function CameraPage() {
  const [mjpegUrl, setMjpegUrl] = useState('http://192.168.1.13:8080/video');
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = () => {
    setError(null);
    if (!mjpegUrl) {
      setError('Enter a valid MJPEG URL');
      return;
    }
    setActive(true);
  };

  const stop = () => setActive(false);

  const mjpegHtml = (url: string) => `<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    html,body,#stream{height:100%;margin:0;background:#000}
    #stream img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block}
  </style>
</head>
<body id="stream">
  <img src="${url}" />
</body>
</html>`;

  return (
    <ThemedView style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <ThemedText type="title">MJPEG Stream</ThemedText>

        <View style={styles.panel}>
          {active ? (
            <WebView
                originWhitelist={["*"]}
                source={{ html: mjpegHtml(mjpegUrl) }}
                style={{ flex: 1, height: '100%' }}
                scalesPageToFit
                mixedContentMode="always"
                javaScriptEnabled
                domStorageEnabled
              />
          ) : (
            <View style={{ flex: 1, backgroundColor: '#000' }} />
          )}
        </View>

        <View style={styles.controls}>
          <TextInput
            value={mjpegUrl}
            onChangeText={setMjpegUrl}
            placeholder="http://PHONE_IP:8080/video"
            style={styles.input}
            autoCapitalize="none"
            keyboardType="url"
          />
          <View style={{ height: 8 }} />
          <Button title={active ? 'Stop stream' : 'Start stream'} onPress={active ? stop : start} />
          {error && <ThemedText type="small" style={styles.error}>{error}</ThemedText>}
        </View>
      </SafeAreaView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  safeArea: { flex: 1, padding: 16 },
  panel: { width: '100%', flex: 1, minHeight: 300, backgroundColor: '#000', borderRadius: 8, overflow: 'hidden' },
  controls: { marginTop: 12 },
  input: { borderWidth: 1, padding: 8, borderRadius: 6, backgroundColor: '#fff' },
  error: { marginTop: 8, color: 'red' },
});
