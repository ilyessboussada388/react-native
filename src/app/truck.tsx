import React, { useState } from 'react';
import { StyleSheet, TextInput, View, Pressable } from 'react-native';
import { ThemedView } from '@/components/themed-view';
import { ThemedText } from '@/components/themed-text';
import { Spacing } from '@/constants/theme';

export default function TruckMetricsPage() {
  const [width, setWidth] = useState('');
  const [height, setHeight] = useState('');
  const [length, setLength] = useState('');

  const parse = (value: string) => {
    const normalized = value.replace(',', '.');
    const n = parseFloat(normalized);
    return Number.isFinite(n) ? n : NaN;
  };

  const w = parse(width);
  const h = parse(height);
  const l = parse(length);
  const volume = Number.isFinite(w) && Number.isFinite(h) && Number.isFinite(l) ? w * h * l : NaN;

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title" style={styles.title}>
        Truck cargo metrics
      </ThemedText>

      <View style={styles.row}>
        <ThemedText>Width</ThemedText>
        <TextInput
          style={styles.input}
          value={width}
          onChangeText={setWidth}
          placeholder="e.g. 2.5"
          keyboardType="numeric"
        />
      </View>

      <View style={styles.row}>
        <ThemedText>Height</ThemedText>
        <TextInput
          style={styles.input}
          value={height}
          onChangeText={setHeight}
          placeholder="e.g. 1.8"
          keyboardType="numeric"
        />
      </View>

      <View style={styles.row}>
        <ThemedText>Length</ThemedText>
        <TextInput
          style={styles.input}
          value={length}
          onChangeText={setLength}
          placeholder="e.g. 6.0"
          keyboardType="numeric"
        />
      </View>

      <View style={styles.resultBox}>
        <ThemedText type="smallBold">Volume</ThemedText>
        {Number.isFinite(volume) ? (
          <ThemedText type="subtitle">{volume.toFixed(3)} (units³)</ThemedText>
        ) : (
          <ThemedText type="small">Enter valid numbers to see volume</ThemedText>
        )}
      </View>

      <Pressable
        style={styles.clearButton}
        onPress={() => {
          setWidth('');
          setHeight('');
          setLength('');
        }}
      >
        <ThemedText type="small">Clear</ThemedText>
      </Pressable>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: Spacing.four,
    gap: Spacing.three,
  },
  title: {
    marginBottom: Spacing.two,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.two,
  },
  input: {
    minWidth: 120,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ccc',
    textAlign: 'right',
  },
  resultBox: {
    marginTop: Spacing.three,
    padding: Spacing.three,
    borderRadius: 8,
  },
  clearButton: {
    marginTop: Spacing.two,
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
  },
});
