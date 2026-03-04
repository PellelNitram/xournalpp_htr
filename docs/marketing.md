# Marketing & Promotion Strategy

Goal: increase GitHub stars by following Andrej Karpathy's approach — go deep into the
project, then speak about it authentically.

---

## Core principle

Every distribution effort should point back to one canonical long-form piece. That piece
should demonstrate genuine technical depth, not just list features. Readers should come
away feeling like they learned something, not just saw an ad.

---

## The anchor content piece

Write a long-form technical blog post on your personal site. Everything else links to it.

**What to cover:**

- The core challenge: bridging a C++ application (Xournal++) with a Python ML pipeline
- HTR model architecture choices and tradeoffs (word detection + text recognition)
- What approaches were tried and failed, and why
- Dataset challenges: what training data exists, what doesn't, how you worked around it
- The integration architecture (how the plugin calls the Python backend)
- The "magic moment" demo — animated GIF or embedded video of handwriting → text
- What's next / open problems

**Tone:** follow Karpathy's style — show your thought process, be honest about what was
hard, lead with the most interesting technical problem, not the feature list.

---

## Distribution channels

### Tier 1 — Highest leverage, do these first

| Platform | Format | Notes |
|---|---|---|
| **Hacker News** (Show HN) | "Show HN: Handwritten text recognition for Xournal++" + link to blog post | Post on a weekday morning, US Eastern time. This audience overlaps perfectly: ML, Linux, open source |
| **Twitter / X thread** | 8–12 tweet thread with demo GIF + link to blog post | Lead with the GIF. No text needed to explain — the demo is self-evident |
| **r/MachineLearning** | Link post with demo GIF in the body | ML practitioners will star if the technical depth is there |

### Tier 2 — Niche but high conversion (existing users, high intent)

| Platform | Notes |
|---|---|
| **Xournal++ GitHub Discussions** | These are users already invested in the app. If there's an open issue/feature request for HTR, close it with a link to this project |
| **r/Xournal** | Same audience, different venue |
| **r/notetaking** | Note-taking enthusiasts are always searching for handwriting → text solutions |
| **r/ObsidianMD** | Similar audience, large community, open to adjacent tooling |
| **r/linux** / **r/kde** | Xournal++ is a native Linux/KDE app — this is the home crowd |
| **r/selfhosted** | Open source + self-hosted ML pipeline angle resonates here |

### Tier 3 — Broader reach

| Platform | Notes |
|---|---|
| **r/opensource** | General open source interest |
| **r/studytips** | Students who handwrite notes are a primary use case |
| **LinkedIn** | ML/research angle; reach academic and industry practitioners |
| **Mastodon** (tech instances) | Strong open source culture; Fosstodon is a good target |
| **Dev.to** | Developer audience, good SEO, cross-post the blog article |

### Writing and discovery platforms

| Platform | Notes |
|---|---|
| **HuggingFace blog** | Already on HF Spaces — reach out to HF team about featuring community projects; they sometimes promote good demos |
| **Papers With Code** | If you have benchmark numbers on IAM or other HTR datasets, submit there for academic visibility |
| **Product Hunt** | Good for a launch moment; treat the HF demo as the "product" |

### Newsletters (submit to, don't post yourself)

| Newsletter | How to submit |
|---|---|
| **TLDR ML** | Community submissions via their site |
| **Import AI** (Jack Clark) | Email or Twitter |
| **Python Weekly** | Submission form on their site |
| **Linux Weekly News** | Submit via lwn.net |

### Video (underrated, low barrier)

- **YouTube Shorts / TikTok**: A 30-second clip of writing something by hand and watching
  it get recognized is inherently shareable. No narration needed — the visual is the story.
- The existing YouTube video can be edited into shorter clips for Shorts/Reels.

---

## Key assets to prepare before posting

1. **Animated GIF**: handwriting on tablet → recognized text appearing. This is the single
   most shareable asset. Embed it in every post.
2. **Long-form blog post**: the canonical piece (see above). Everything links here.
3. **One-liner description**: a crisp sentence for HN/Reddit titles.
   Example: *"Open source plugin that uses ML to recognize handwriting in Xournal++ notes."*
4. **HuggingFace demo link**: the zero-friction try-it-now path. Always include this.

---

## Timing

- Do all Tier 1 posts within the same week to create a spike
- HN first (highest potential upside), then Twitter, then Reddit
- Follow up 2–3 weeks later on Tier 2 communities with any traction numbers ("posted to HN,
  here's what I learned building it")

---

## What not to do

- Don't post everywhere at once without the anchor content piece ready
- Don't lead with the feature list — lead with the hardest problem you solved
- Don't ignore the Xournal++ community itself (Tier 2) — they are the highest-intent users
