const AUTO_SLIDE_MS = 30000;
const BILL_TRANSITION_MS = 360;

// Backend API base URL. In local development where the frontend and
// backend are served from the same origin, leave this as an empty
// string so relative paths like "/dashboard" are used. In
// production (e.g. Vercel frontend + Render backend), set this to
// your Render URL, for example:
// const API_BASE = "https://your-backend.onrender.com";
//
// To make this file safe to push before you know the URL, we use a
// small heuristic: when running on Vercel we expect the hostname to
// end with "vercel.app" and route calls to the Render API; otherwise
// we default to same-origin.
const API_BASE = window.location.hostname.endsWith("vercel.app")
  ? "https://ai-legislative-analyzer-81x2.onrender.com"
  : "";

const billSlideEl = document.getElementById("bill-slide");
const billCounterEl = document.getElementById("bill-counter");
const billTitleEl = document.getElementById("bill-title");
const purposeTextEl = document.getElementById("purpose-text");
const keyPointsTextEl = document.getElementById("keypoints-text");
const impactTextEl = document.getElementById("impact-text");
const openPdfBtnEl = document.getElementById("open-pdf-btn");

const prevBillBtnEl = document.getElementById("prev-bill-btn");
const nextBillBtnEl = document.getElementById("next-bill-btn");

const chatToggleEl = document.getElementById("chat-toggle");
const chatBackdropEl = document.getElementById("chat-backdrop");
const chatPanelEl = document.getElementById("chat-panel");
const chatCloseEl = document.getElementById("chat-close");
const chatMessagesEl = document.getElementById("chat-messages");
const chatFormEl = document.getElementById("chat-form");
const chatInputEl = document.getElementById("chat-input");
const chatSendEl = document.getElementById("chat-send");

const detailsBackdropEl = document.getElementById("details-backdrop");
const detailsModalEl = document.getElementById("details-modal");
const detailsTitleEl = document.getElementById("details-title");
const detailsBodyEl = document.getElementById("details-body");
const detailsCloseEl = document.getElementById("details-close");

let bills = [];
let currentBillIndex = 0;
let autoSlideTimer = null;
let autoSlidePausedByUser = false;
let isAnimating = false;
let currentSummaryParts = null;
let activeChatPdfUrl = null;

function normalizeLineText(line) {
  return String(line || "")
    .replace(/^[-*\u2022]\s*/, "")
    .replace(/\*\*/g, "")
    .trim();
}

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function cleanSummaryText(text) {
  return normalizeLineText(text)
    .replace(/^here'?s\s+a\s+summary\s+of[^:]*:\s*/i, "")
    .replace(/^summary\s*:\s*/i, "")
    .trim();
}

function cleanBillTitle(rawTitle) {
  const original = String(rawTitle || "Untitled bill").trim();
  let cleaned = original
    .replace(/\.pdf$/i, "")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const subtitleBits = [];
  const bracketRegex = /\(([^)]+)\)|\[([^\]]+)\]/g;
  cleaned = cleaned.replace(bracketRegex, (_, round, square) => {
    const text = (round || square || "").trim();
    if (text) subtitleBits.push(text);
    return " ";
  });

  cleaned = cleaned.replace(/\s+/g, " ").trim();

  const colonSplit = cleaned.split(/\s[-:]\s/);
  let mainTitle = cleaned;
  if (colonSplit.length > 1) {
    mainTitle = colonSplit[0].trim();
    subtitleBits.push(colonSplit.slice(1).join(" - ").trim());
  }

  mainTitle = mainTitle
    .replace(/\b(Bill|Act)\b\s*$/i, (match) => match.charAt(0).toUpperCase() + match.slice(1).toLowerCase())
    .trim();

  const subtitle = subtitleBits
    .map((part) => cleanSummaryText(part))
    .filter(Boolean)
    .join(" | ");

  return {
    mainTitle: mainTitle || "Untitled bill",
    subtitle,
  };
}

function inferTitleFromSummary(summary) {
  const text = cleanSummaryText(summary);
  if (!text) {
    return "";
  }

  const patterns = [
    /summary\s+of\s+the\s+(.+?)(?:\.|:|,\s*\d{4}|$)/i,
    /about\s+the\s+(.+?)(?:\.|:|,\s*\d{4}|$)/i,
    /^(?:the\s+)?(.+?)(?:\s+is\s+|\s+proposes\s+|\s+aims\s+|\.|:|$)/i,
  ];

  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (!match || !match[1]) {
      continue;
    }

    const candidate = cleanSummaryText(match[1])
      .replace(/\b(here'?s|summary|bill)\b/gi, "")
      .replace(/\s+/g, " ")
      .trim();

    if (candidate.length >= 10) {
      return candidate;
    }
  }

  return "";
}

function extractSentences(text) {
  const cleaned = cleanSummaryText(text);
  if (!cleaned) {
    return [];
  }

  const normalized = cleaned.replace(/\s*\n+\s*/g, " ");

  let parts = normalized
    .split(/(?<=[.!?])\s+(?=[A-Z0-9])/)
    .map((p) => cleanSummaryText(p))
    .filter(Boolean);

  if (parts.length === 1) {
    parts = normalized
      .split(/;\s+|\u2022\s+|-\s+(?=[A-Z0-9])/)
      .map((p) => cleanSummaryText(p))
      .filter(Boolean);
  }

  const expanded = [];
  for (const sentence of parts) {
    if (sentence.length <= 260) {
      expanded.push(sentence);
      continue;
    }

    const clauses = sentence
      .split(/;\s+|,\s+(?=[A-Z])/)
      .map((c) => cleanSummaryText(c))
      .filter(Boolean);

    if (clauses.length > 1) {
      expanded.push(...clauses);
    } else {
      expanded.push(sentence);
    }
  }

  return expanded;
}

function makePreviewSentence(sentence, maxLength = 160) {
  let value = cleanSummaryText(sentence);
  if (!value) {
    return "";
  }

  value = value.replace(/^\w[\w\s]{1,30}:\s*/i, "");

  if (value.length <= maxLength) {
    return value;
  }

  const clauseSplit = value
    .split(/;\s+|,\s+(?=[A-Z])/)
    .map((part) => cleanSummaryText(part))
    .filter(Boolean);

  if (clauseSplit.length > 1 && clauseSplit[0].length >= 40 && clauseSplit[0].length <= maxLength) {
    return clauseSplit[0];
  }

  const softLimit = Math.max(60, maxLength - 30);
  let cutIndex = value.lastIndexOf(" ", maxLength);
  if (cutIndex < softLimit) {
    cutIndex = value.indexOf(" ", softLimit);
    if (cutIndex === -1 || cutIndex > maxLength) {
      cutIndex = maxLength;
    }
  }

  const trimmed = value.slice(0, cutIndex).trimEnd().replace(/[.,;:-]*$/, "");
  return `${trimmed}...`;
}

function extractLabeledSection(summary, labelCandidates) {
  for (const label of labelCandidates) {
    const escapedLabel = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const pattern = new RegExp(
      `(?:\\*\\*)?${escapedLabel}(?:\\*\\*)?\\s*:\\s*([\\s\\S]*?)(?=(?:\\*\\*?[A-Za-z][A-Za-z\\s]{2,40}\\*\\*?\\s*:)|$)`,
      "i"
    );

    const match = summary.match(pattern);
    if (match && match[1]) {
      return cleanSummaryText(match[1]);
    }
  }

  return "";
}

function splitIntoReadablePoints(text) {
  const cleaned = cleanSummaryText(text);
  if (!cleaned) {
    return [];
  }

  const sentenceParts = cleaned
    .split(/(?<=[.!?])\s+(?=[A-Z])/)
    .map((part) => cleanSummaryText(part))
    .filter(Boolean);

  if (sentenceParts.length > 1) {
    return sentenceParts;
  }

  const commaParts = cleaned
    .split(/,\s+(?=[A-Za-z])/)
    .map((part) => cleanSummaryText(part))
    .filter(Boolean);

  if (commaParts.length > 1) {
    return commaParts;
  }

  return [cleaned];
}

function buildCardPoints(text) {
  const sentences = extractSentences(text);

  if (!sentences.length) {
    return ["Not enough details available yet."];
  }

  const deduped = [];
  const seen = new Set();

  for (let sentence of sentences) {
    const key = sentence.toLowerCase();
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);

    sentence = sentence.replace(/^\w[\w\s]{1,30}:\s*/i, "");
    sentence = cleanSummaryText(sentence);
    if (!sentence) {
      continue;
    }

    deduped.push(sentence);
  }

  const MAX_POINTS = 2;
  const trimmed = deduped.slice(0, MAX_POINTS).map((s) => makePreviewSentence(s)).filter(Boolean);

  return trimmed.length ? trimmed : ["Not enough details available yet."];
}

function formatCardContent(text) {
  const points = buildCardPoints(text);
  if (!points.length) {
    return "<ul><li>Not enough details available yet.</li></ul>";
  }

  const items = points
    .map((point) => `<li>${escapeHtml(point)}</li>`)
    .join("");

  return `<ul>${items}</ul>`;
}

function splitSummaryToThreeParts(summary) {
  const raw = String(summary || "").trim().replace(/\*\*/g, "");
  if (!raw) {
    return {
      purpose: "No summary available for this bill yet.",
      keyPoints: "No key points found.",
      impact: "No impact details available.",
      subtitle: "Summary will appear when processing is complete.",
    };
  }

  const lines = raw
    .split(/\r?\n/)
    .map(normalizeLineText)
    .filter(Boolean);

  const labeledPurpose = extractLabeledSection(raw, ["Purpose", "Purpose of the Bill", "Objective"]);
  const labeledKeyPoints = extractLabeledSection(raw, ["Key Points", "Key Provisions", "Main Provisions", "What Changes"]);
  const labeledImpact = extractLabeledSection(raw, ["Impact", "Who is Affected", "Expected Impact", "Implications"]);

  const sentenceChunks = lines.length
    ? lines
    : raw
        .split(/(?<=[.!?])\s+/)
        .map((part) => part.trim())
        .filter(Boolean);

  const total = sentenceChunks.length;
  const partSize = Math.max(1, Math.ceil(total / 3));

  const purpose = sentenceChunks.slice(0, partSize).join(" ");
  const keyPoints = sentenceChunks.slice(partSize, partSize * 2).join(" ");
  const impact = sentenceChunks.slice(partSize * 2).join(" ");

  const subtitleCandidate = cleanSummaryText(sentenceChunks[0] || "");

  return {
    subtitle: subtitleCandidate || "AI summary generated from bill text",
    purpose: labeledPurpose || purpose || "No purpose information available.",
    keyPoints: labeledKeyPoints || keyPoints || "No key points available.",
    impact: labeledImpact || impact || "No impact analysis available.",
  };
}

function updatePdfButton(pdfUrl) {
  if (pdfUrl) {
    openPdfBtnEl.href = pdfUrl;
    openPdfBtnEl.style.display = "inline-flex";
  } else {
    openPdfBtnEl.href = "#";
    openPdfBtnEl.style.display = "none";
  }
}

function renderCurrentBill() {
  if (!bills.length) {
    billCounterEl.textContent = "0 bills loaded";
    billTitleEl.textContent = "No bills loaded yet";
    purposeTextEl.innerHTML = "<ul><li>No summary available for this bill yet.</li></ul>";
    keyPointsTextEl.innerHTML = "<ul><li>No key points found.</li></ul>";
    impactTextEl.innerHTML = "<ul><li>No impact details available.</li></ul>";
    updatePdfButton(null);
    return;
  }

  const bill = bills[currentBillIndex];
  const titleParts = cleanBillTitle(bill.title);
  const summaryParts = splitSummaryToThreeParts(bill.summary);
  currentSummaryParts = summaryParts;
  const inferredSummaryTitle = inferTitleFromSummary(bill.summary);

  let displayTitle = titleParts.mainTitle;
  if ((displayTitle.toLowerCase() === "untitled bill" || displayTitle.length < 8) && inferredSummaryTitle) {
    displayTitle = inferredSummaryTitle;
  }

  billCounterEl.textContent = `${bills.length} bill${bills.length === 1 ? "" : "s"} loaded`;
  billTitleEl.textContent = displayTitle;
  billTitleEl.title = titleParts.mainTitle;

  purposeTextEl.innerHTML = formatCardContent(summaryParts.purpose);
  keyPointsTextEl.innerHTML = formatCardContent(summaryParts.keyPoints);
  impactTextEl.innerHTML = formatCardContent(summaryParts.impact);

  updatePdfButton(bill.pdf_url || null);
}

function animateToBill(nextIndex, direction) {
  if (isAnimating || !bills.length || nextIndex === currentBillIndex) {
    return;
  }

  isAnimating = true;
  billSlideEl.classList.add(direction === "next" ? "slide-out-left" : "slide-out-right");

  window.setTimeout(() => {
    billSlideEl.classList.remove("slide-out-left", "slide-out-right");
    billSlideEl.classList.add(direction === "next" ? "slide-in-left" : "slide-in-right");

    currentBillIndex = nextIndex;
    renderCurrentBill();

    requestAnimationFrame(() => {
      billSlideEl.classList.remove("slide-in-left", "slide-in-right");
      window.setTimeout(() => {
        isAnimating = false;
      }, BILL_TRANSITION_MS);
    });
  }, 210);
}

function nextBill() {
  if (!bills.length) return;
  const nextIndex = (currentBillIndex + 1) % bills.length;
  animateToBill(nextIndex, "next");
}

function previousBill() {
  if (!bills.length) return;
  const previousIndex = (currentBillIndex - 1 + bills.length) % bills.length;
  animateToBill(previousIndex, "previous");
}

function stopAutoSlide() {
  if (autoSlideTimer) {
    clearInterval(autoSlideTimer);
    autoSlideTimer = null;
  }
}

function startAutoSlide() {
  stopAutoSlide();
  autoSlideTimer = setInterval(() => {
    if (!autoSlidePausedByUser) {
      nextBill();
    }
  }, AUTO_SLIDE_MS);
}

function pauseAutoSlideByUser() {
  autoSlidePausedByUser = true;
  stopAutoSlide();
}

async function fetchBills() {
  try {
    const response = await fetch(`${API_BASE}/dashboard`);
    if (!response.ok) {
      throw new Error(`Dashboard request failed: ${response.status}`);
    }

    const payload = await response.json();
    if (!Array.isArray(payload)) {
      throw new Error("Unexpected /dashboard response format");
    }

    const currentPdfUrl = bills[currentBillIndex]?.pdf_url || null;
    bills = payload;

    if (!bills.length) {
      currentBillIndex = 0;
      renderCurrentBill();
      return;
    }

    if (currentPdfUrl) {
      const sameBillIndex = bills.findIndex((bill) => bill.pdf_url === currentPdfUrl);
      currentBillIndex = sameBillIndex >= 0 ? sameBillIndex : 0;
    } else if (currentBillIndex >= bills.length) {
      currentBillIndex = 0;
    }

    renderCurrentBill();
  } catch (error) {
    console.error(error);
  }
}
function formatBotAnswerText(rawText) {
  const raw = String(rawText || "");
  if (!raw.trim()) {
    return "I could not generate an answer for that yet.";
  }

  let value = raw.replace(/\r\n/g, "\n");

  // Remove fenced code blocks
  value = value.replace(/```[\s\S]*?```/g, "");
  // Strip markdown headings (e.g., ## Title)
  value = value.replace(/^\s{0,3}#{1,6}\s*/gm, "");
  // Strip blockquote markers
  value = value.replace(/^\s*>\s*/gm, "");
  // Remove list bullets and numbered list prefixes
  value = value.replace(/^\s*[-*+•]\s+/gm, "");
  value = value.replace(/^\s*\d+[\.\)]\s+/gm, "");
  // Remove bold/italic markers (**text**, *text*, __text__, _text_)
  value = value.replace(/(\*\*|__)(.*?)\1/g, "$2");
  value = value.replace(/(\*|_)(.*?)\1/g, "$2");
  // Trim trailing whitespace per line
  value = value.replace(/[ \t]+$/gm, "");
  // Collapse 3+ blank lines down to 2
  value = value.replace(/\n{3,}/g, "\n\n");

  return value.trim();
}

function appendMessage(role, text) {
  const messageEl = document.createElement("div");
  messageEl.className = `chat-message ${role}`;

  if (role === "bot") {
    const cleaned = formatBotAnswerText(text);
    const paragraphs = cleaned.split(/\n{2,}/).filter(Boolean);

    if (paragraphs.length) {
      paragraphs.forEach((para) => {
        const p = document.createElement("p");
        p.textContent = para;
        messageEl.appendChild(p);
      });
    } else {
      messageEl.textContent = cleaned;
    }
  } else {
    messageEl.textContent = text;
  }

  chatMessagesEl.appendChild(messageEl);
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
}

function openChat() {
  chatBackdropEl.classList.add("open");
  chatBackdropEl.setAttribute("aria-hidden", "false");
  chatPanelEl.classList.add("open");
  chatPanelEl.setAttribute("aria-hidden", "false");
  chatInputEl.focus();
}

function closeChat() {
  chatBackdropEl.classList.remove("open");
  chatBackdropEl.setAttribute("aria-hidden", "true");
  chatPanelEl.classList.remove("open");
  chatPanelEl.setAttribute("aria-hidden", "true");
}

async function submitChatQuestion(event) {
  event.preventDefault();

  const query = chatInputEl.value.trim();
  if (!query) {
    return;
  }

  appendMessage("user", query);
  chatInputEl.value = "";
  chatInputEl.focus();

  chatSendEl.disabled = true;

  const loadingEl = document.createElement("div");
  loadingEl.className = "chat-message bot thinking";
  loadingEl.textContent = "Thinking...";
  chatMessagesEl.appendChild(loadingEl);
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;

  try {
    let url = `${API_BASE}/ask?query=${encodeURIComponent(query)}`;
    if (activeChatPdfUrl) {
      url += `&pdf_url=${encodeURIComponent(activeChatPdfUrl)}`;
    }

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Ask request failed: ${response.status}`);
    }

    const payload = await response.json();
    loadingEl.remove();
    if (payload.pdf_url) {
      activeChatPdfUrl = payload.pdf_url;
    }

    appendMessage("bot", payload.answer || "I could not generate an answer for that yet.");
  } catch (error) {
    console.error(error);
    loadingEl.remove();
    appendMessage("bot", "I hit an error while answering. Please try again.");
  } finally {
    chatSendEl.disabled = false;
  }
}

function openDetails(sectionKey) {
  if (!currentSummaryParts) {
    return;
  }

  let title;
  let fullText;

  if (sectionKey === "purpose") {
    title = "Purpose";
    fullText = currentSummaryParts.purpose;
  } else if (sectionKey === "keyPoints") {
    title = "Key Points";
    fullText = currentSummaryParts.keyPoints;
  } else if (sectionKey === "impact") {
    title = "Impact";
    fullText = currentSummaryParts.impact;
  } else {
    return;
  }

  const sentences = extractSentences(fullText);
  let bodyHtml;

  if (sentences.length) {
    const items = sentences.map((s) => `<li>${escapeHtml(s)}</li>`).join("");
    bodyHtml = `<ul>${items}</ul>`;
  } else {
    bodyHtml = `<p>${escapeHtml(cleanSummaryText(fullText) || "No details available.")}</p>`;
  }

  detailsTitleEl.textContent = title;
  detailsBodyEl.innerHTML = bodyHtml;

  detailsBackdropEl.classList.add("open");
  detailsBackdropEl.setAttribute("aria-hidden", "false");
  detailsModalEl.classList.add("open");
  detailsModalEl.setAttribute("aria-hidden", "false");
}

function closeDetails() {
  detailsBackdropEl.classList.remove("open");
  detailsBackdropEl.setAttribute("aria-hidden", "true");
  detailsModalEl.classList.remove("open");
  detailsModalEl.setAttribute("aria-hidden", "true");
}

prevBillBtnEl.addEventListener("click", () => {
  pauseAutoSlideByUser();
  previousBill();
});

nextBillBtnEl.addEventListener("click", () => {
  pauseAutoSlideByUser();
  nextBill();
});

chatToggleEl.addEventListener("click", () => {
  if (chatPanelEl.classList.contains("open")) {
    closeChat();
  } else {
    openChat();
  }
});

chatCloseEl.addEventListener("click", closeChat);
chatBackdropEl.addEventListener("click", closeChat);
chatFormEl.addEventListener("submit", submitChatQuestion);

document.querySelectorAll(".view-more-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const section = btn.getAttribute("data-section");
    openDetails(section);
  });
});

detailsBackdropEl.addEventListener("click", closeDetails);
detailsCloseEl.addEventListener("click", closeDetails);

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    if (chatPanelEl.classList.contains("open")) {
      closeChat();
    }
    if (detailsModalEl.classList.contains("open")) {
      closeDetails();
    }
  }
});

// Keep bill list fresh without changing endpoints or introducing new APIs.
setInterval(fetchBills, AUTO_SLIDE_MS);

fetchBills();
startAutoSlide();
