const btn = document.getElementById("recBtn");
const status = document.getElementById("status");
const result = document.getElementById("result");

let recorder, chunks = [], recording = false;

btn.onclick = async () => {
  if (!recording) {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    chunks = [];

    recorder.ondataavailable = e => chunks.push(e.data);
    recorder.start();

    recording = true;
    btn.innerText = "⏹️ Parar gravação";
    status.innerText = "🎙️ Gravando...";
    result.innerText = "";
  } else {
    recorder.stop();
    recording = false;

    btn.innerText = "🎤 Gravar áudio";
    status.innerText = "⏳ Convertendo áudio...";

    recorder.onstop = async () => {
      const webmBlob = new Blob(chunks, { type: "audio/webm" });
      const arrayBuffer = await webmBlob.arrayBuffer();
      const audioCtx = new AudioContext();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

      const wavBlob = bufferToWave(audioBuffer, audioBuffer.length);
      const form = new FormData();
      form.append("file", wavBlob, "voice.wav");

      status.innerText = "⏳ Enviando áudio...";

      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: form
      });

      const data = await res.json();
      result.innerText = `Emoção detectada: ${data.emotion}`;
      status.innerText = "";
    };
  }
};

function bufferToWave(abuffer, len) {
  const numOfChan = abuffer.numberOfChannels,
    length = len * numOfChan * 2 + 44,
    buffer = new ArrayBuffer(length),
    view = new DataView(buffer),
    channels = [],
    sampleRate = abuffer.sampleRate;

  let offset = 0;

  function writeString(s) {
    for (let i = 0; i < s.length; i++) view.setUint8(offset++, s.charCodeAt(i));
  }

  writeString("RIFF");
  view.setUint32(offset, length - 8, true); offset += 4;
  writeString("WAVE");
  writeString("fmt ");
  view.setUint32(offset, 16, true); offset += 4;
  view.setUint16(offset, 1, true); offset += 2;
  view.setUint16(offset, numOfChan, true); offset += 2;
  view.setUint32(offset, sampleRate, true); offset += 4;
  view.setUint32(offset, sampleRate * 2 * numOfChan, true); offset += 4;
  view.setUint16(offset, numOfChan * 2, true); offset += 2;
  view.setUint16(offset, 16, true); offset += 2;
  writeString("data");
  view.setUint32(offset, length - offset - 4, true); offset += 4;

  const interleaved = interleave(abuffer);
  for (let i = 0; i < interleaved.length; i++, offset += 2) {
    view.setInt16(offset, interleaved[i] * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function interleave(input) {
  const channels = [];
  for (let i = 0; i < input.numberOfChannels; i++) channels.push(input.getChannelData(i));
  const length = channels[0].length;
  const result = new Float32Array(length * channels.length);
  let index = 0;
  for (let i = 0; i < length; i++)
    for (let ch = 0; ch < channels.length; ch++) result[index++] = channels[ch][i];
  return result;
}
