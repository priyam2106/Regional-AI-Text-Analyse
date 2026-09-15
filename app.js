const samples = {
  hindi: "कृत्रिम बुद्धिमत्ता आज शिक्षा, स्वास्थ्य और कृषि जैसे क्षेत्रों में तेज़ी से उपयोग की जा रही है। यह तकनीक बड़े डेटा का विश्लेषण करके उपयोगी सुझाव दे सकती है, जिससे निर्णय लेने की प्रक्रिया बेहतर बनती है। हालांकि, इसके उपयोग में पारदर्शिता और मानवीय निगरानी आवश्यक है।",
  tamil: "செயற்கை நுண்ணறிவு இன்று கல்வி மற்றும் மருத்துவத் துறைகளில் குறிப்பிடத்தக்க மாற்றங்களை உருவாக்குகிறது. தரவுகளை விரைவாக ஆய்வு செய்து பயனுள்ள தகவல்களை வழங்கும் திறன் இதற்கு உள்ளது. இருப்பினும், இந்தத் தொழில்நுட்பத்தை பொறுப்புடன் பயன்படுத்துவது மிகவும் முக்கியம்.",
  english: "Artificial intelligence is changing the way people work and learn. It can process large amounts of information quickly and offer useful suggestions. At the same time, responsible use, clear oversight, and careful fact-checking remain essential."
};
const textInput = document.querySelector('#textInput');
const charCount = document.querySelector('#charCount');
const results = document.querySelector('#results');
const scoreRing = document.querySelector('#scoreRing');

textInput.addEventListener('input', () => charCount.textContent = `${textInput.value.length.toLocaleString()} / 5,000`);
document.querySelector('#clearText').addEventListener('click', () => { textInput.value = ''; textInput.dispatchEvent(new Event('input')); textInput.focus(); });
document.querySelectorAll('[data-sample]').forEach(button => button.addEventListener('click', () => { textInput.value = samples[button.dataset.sample]; textInput.dispatchEvent(new Event('input')); }));
document.querySelector('#newAnalysis').addEventListener('click', () => { results.classList.add('is-hidden'); document.querySelector('#analyze').scrollIntoView({behavior:'smooth'}); textInput.focus(); });

function detectLanguage(text) {
  if (/\p{Script=Devanagari}/u.test(text)) return 'Hindi / Devanagari';
  if (/\p{Script=Bengali}/u.test(text)) return 'Bengali';
  if (/\p{Script=Tamil}/u.test(text)) return 'Tamil';
  if (/\p{Script=Telugu}/u.test(text)) return 'Telugu';
  if (/\p{Script=Gujarati}/u.test(text)) return 'Gujarati';
  return 'English';
}
function analyze(text) {
  const words = text.trim().split(/\s+/).filter(Boolean);
  const sentences = text.split(/[.!?।]/).map(s => s.trim()).filter(Boolean);
  const uniqueRatio = new Set(words.map(w => w.toLowerCase().replace(/[^\p{L}]/gu, ''))).size / Math.max(words.length, 1);
  const lengths = sentences.map(s => s.split(/\s+/).length);
  const variation = lengths.length > 1 ? Math.sqrt(lengths.reduce((sum,n) => sum + (n - lengths.reduce((a,b)=>a+b,0)/lengths.length) ** 2,0) / lengths.length) : 0;
  const connectors = (text.match(/\b(however|therefore|moreover|additionally|however|हालांकि|इसके अलावा|இருப்பினும்|மேலும்)\b/giu) || []).length;
  const base = 42 + (1 - uniqueRatio) * 30 + Math.max(0, 8 - variation) * 1.5 + Math.min(connectors * 4, 10);
  const score = Math.max(12, Math.min(91, Math.round(base + (words.length % 13) - 6)));
  return { score, language: detectLanguage(text), signals: [
    ['Sentence regularity', Math.round(Math.max(18, Math.min(92, 72 - variation * 5)))],
    ['Vocabulary predictability', Math.round(Math.max(15, Math.min(94, (1 - uniqueRatio) * 100)))],
    ['Structural consistency', Math.round(Math.max(20, Math.min(90, 48 + connectors * 8 + words.length / 18)))],
  ]};
}
async function trainedEstimate(text, language) {
  const response = await fetch('/api/analyze', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text, language})
  });
  if (!response.ok) throw new Error('Trained model unavailable');
  return response.json();
}
document.querySelector('#analyzeButton').addEventListener('click', async () => {
  const text = textInput.value.trim();
  if (text.length < 30) { textInput.focus(); textInput.placeholder = 'Please enter at least a sentence (30 characters) for a meaningful check.'; return; }
  const button = document.querySelector('#analyzeButton');
  button.disabled = true; button.textContent = 'Analyzing…';
  const outcome = analyze(text); const selected = document.querySelector('#language').value;
  const language = selected === 'auto' ? outcome.language : selected;
  try {
    const prediction = await trainedEstimate(text, language);
    outcome.score = Math.round(prediction.ai_probability);
    outcome.mode = 'TRAINED HINDI MODEL';
  } catch (_) {
    outcome.mode = 'LOCAL DEMO ESTIMATE';
  } finally {
    button.disabled = false; button.innerHTML = 'Analyze text <span>→</span>';
  }
  document.querySelector('#score').textContent = outcome.score;
  scoreRing.style.setProperty('--progress', outcome.score);
  document.querySelector('#resultKicker').textContent = outcome.mode;
  const verdict = outcome.score >= 70 ? ['Likely AI-assisted', 'This passage has several patterns often found in generated text.'] : outcome.score >= 45 ? ['Mixed signals', 'The writing contains both human and AI-associated patterns.'] : ['Likely human-written', 'The passage shows natural variation more often seen in human writing.'];
  document.querySelector('#verdict').textContent = verdict[0];
  document.querySelector('#verdictDescription').textContent = verdict[1];
  document.querySelector('#detectedLanguage').textContent = language;
  document.querySelector('#signals').innerHTML = outcome.signals.map(([name, value]) => `<div class="signal"><label>${name}</label><div class="track"><div class="fill" style="width:${value}%"></div></div><b>${value}%</b></div>`).join('');
  results.classList.remove('is-hidden');
  setTimeout(() => document.querySelectorAll('.fill').forEach(el => el.style.width = el.style.width), 50);
  results.scrollIntoView({behavior:'smooth', block:'start'});
});
