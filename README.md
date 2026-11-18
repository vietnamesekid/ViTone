# ViTone

**An open conversational speech model for Vietnamese.** Speech that talks with you, not text read aloud.

> **Status: Phase 0, data preparation and licensing.** There is no code or model yet. This README describes what we are building and why.

---

## Why ViTone

Vietnamese text-to-speech has improved quickly. Open models such as VieNeu-TTS, F5-TTS-Vietnamese and XTTS-v2-Vietnamese now produce clean, clonable voices. Almost all of them, however, are trained on **read speech**: audiobooks, news and scripted recordings. They sound like someone reading.

Real conversation sounds different. People:

- use fillers and hesitations (*ừ, à, ờm, kiểu, thì…*);
- laugh, sigh and breathe;
- give short backchannels while the other person talks (*dạ, ừ, đúng rồi*);
- take turns with natural timing, sometimes overlapping;
- switch between Vietnamese and English mid-sentence;
- change tone and pace according to what the other person just said.

As far as we know, no open Vietnamese model yet has all three of the following:

1. **Context awareness.** It conditions on the audio of previous turns, not only on the current sentence.
2. **Real conversational data.** It learns from spontaneous two-person dialogue, not read speech.
3. **Full openness.** Code, weights and the data pipeline are all open.

ViTone aims to fill that gap.

## Goals

| Goal | What it means |
|---|---|
| **Natural Vietnamese conversation** | Fillers, laughter, backchannels and turn-taking come out naturally, not as tags pasted onto read speech |
| **Context-aware generation** | Each turn is generated from the text *and* the audio history of the dialogue |
| **Two-speaker dialogue** | Podcast-style conversations generated with consistent voices over long contexts |
| **Correct tones** | All six Vietnamese tones stay correct even in fast, casual speech |
| **Vietnamese–English code-switching** | English words inside Vietnamese sentences are pronounced naturally |
| **Regional variety** | Northern, Central and Southern accents (long-term) |
| **Open and reproducible** | Apache-2.0 code and weights, plus a documented, reproducible data pipeline |

**Non-goals:** cloning real people's voices, and shipping preset voices of any real host or guest.

## Approach

### Base model

Our primary candidate is **[Sesame CSM-1B](https://huggingface.co/sesame/csm-1b)** (Apache-2.0). It has a Llama-style backbone of about 1B parameters and a small audio decoder that produces Mimi codec tokens. It is designed to condition on conversational context, both text and audio of earlier turns, which is exactly the capability we need. CSM-1B is optimized for English, so Vietnamese requires substantial fine-tuning.

We will also benchmark other open dialogue-capable architectures, such as Dia, Higgs Audio V2, VibeVoice and SoulX-Podcast, before committing.

### Training (planned)

1. **Vietnamese adaptation:** teach the base model Vietnamese phonetics and tones on larger single-speaker data.
2. **Conversational fine-tuning:** train on multi-turn dialogue windows with speaker tags and audio context.
3. **Speaker balancing:** cap per-speaker hours so the model does not collapse into a single dominant voice.

## Evaluation (planned)

| Aspect | Metric |
|---|---|
| Intelligibility | CER from a Vietnamese ASR model |
| Tone accuracy | Tone error rate on a targeted test set |
| Speaker consistency | Speaker-embedding similarity across turns |
| Audio quality | Automatic MOS predictors, plus listening tests |
| Conversational naturalness | Human CMOS against read-speech baselines, judged on turn-taking, fillers and prosody given context |

Test sets are split **by episode and by speaker**, so evaluation always uses unseen voices.

## Roadmap

- [ ] **Phase 0: Data licensing and pilot.** Secure permissions, process a few pilot episodes, and measure diarization and ASR error.
- [ ] **Phase 1: Data pipeline.** Release the open-source pipeline and process the full licensed corpus.
- [ ] **Phase 2: Verbatim ASR.** Fine-tune ASR on hand-labeled verbatim data and re-transcribe.
- [ ] **Phase 3: Model.** Vietnamese adaptation and conversational fine-tuning.
- [ ] **Phase 4: Release.** Weights, evaluation suite, demo and documentation.

## Responsible use

Conversational speech models are powerful, and voices belong to people.

- Do **not** use ViTone to imitate real individuals without their explicit consent.
- Do **not** use it for fraud, scams, misinformation or deceptive content.
- Released models will not include voices of any real person from the training data.
- We will explore audio watermarking for generated speech.

## License

The **code and model weights** are licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attributions.

The **training data is not covered** by this license. Third-party audio remains the property of its owners and is not redistributed by this project.

## Contributing

The project is at an early stage. Ideas, data-source suggestions (with licensing in mind), and help with Vietnamese text normalization, ASR or evaluation are all welcome. Please open an issue to start a discussion.
