"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  app: $("app"), list: $("chatList"), title: $("chatTitle"), thread: $("thread"), messages: $("messages"),
  input: $("input"), send: $("send"), useWeb: $("useWeb"), composer: $("composer"),
  newChat: $("newChat"), menu: $("menu"), scrim: $("scrim"),
  summaryBtn: $("summaryBtn"), summaryPanel: $("summaryPanel"), summaryBody: $("summaryBody"),
  summaryClose: $("summaryClose"), summaryScrim: $("summaryScrim"),
};
const state = { chatId: null, busy: false, chats: [] };

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// ---------- the "Use web tool" choice is remembered in this browser only ----------
function loadPreference() {
  try { els.useWeb.checked = localStorage.getItem("useWeb") === "1"; } catch { /* storage blocked: start unticked */ }
}
function savePreference() {
  try { localStorage.setItem("useWeb", els.useWeb.checked ? "1" : "0"); } catch { /* ignore */ }
}

// ---------- backend ----------
async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* body was not JSON */ }
    throw new Error(typeof detail === "string" ? detail : "The request was rejected");
  }
  return res;
}

// ---------- markdown ----------
if (window.DOMPurify) {
  // Links in answers open in a new tab
  DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "A") {
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noopener noreferrer");
    }
  });
}

function renderMarkdown(text) {
  const box = el("div", "body");
  if (!window.marked || !window.DOMPurify) {  // libraries did not load (offline): show the plain text
    box.style.whiteSpace = "pre-wrap";
    box.textContent = text;
    return box;
  }
  box.innerHTML = DOMPurify.sanitize(marked.parse(text, { gfm: true, breaks: true }));
  box.querySelectorAll("pre").forEach((pre) => {
    const code = pre.querySelector("code");
    if (code && window.hljs) window.hljs.highlightElement(code);
    const wrap = el("div", "codeblock");
    const copy = el("button", "copy", "Copy");
    copy.type = "button";
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText((code || pre).textContent);
        copy.textContent = "Copied";
      } catch {
        copy.textContent = "Press Ctrl+C";
      }
      setTimeout(() => { copy.textContent = "Copy"; }, 1500);
    });
    pre.replaceWith(wrap);
    wrap.append(pre, copy);
  });
  box.querySelectorAll("table").forEach((table) => {
    const wrap = el("div", "table-wrap");
    table.replaceWith(wrap);
    wrap.append(table);
  });
  return box;
}

// ---------- thread ----------
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function scrollToEnd({ smooth = true } = {}) {
  els.thread.scrollTo({
    top: els.thread.scrollHeight,
    behavior: smooth && !prefersReducedMotion.matches ? "smooth" : "auto",
  });
}

// Keeps the view on the bottom while content is still growing (an animation), one frame at a time
function followBottomFor(ms) {
  const end = performance.now() + ms;
  const tick = () => {
    els.thread.scrollTop = els.thread.scrollHeight;
    if (performance.now() < end) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// A long answer is shown from its first line, a short one from its last
function revealReply(row) {
  if (row.offsetHeight > els.thread.clientHeight) {
    els.thread.scrollTo({ top: row.offsetTop - 16, behavior: prefersReducedMotion.matches ? "auto" : "smooth" });
  } else {
    followBottomFor(prefersReducedMotion.matches ? 0 : 400);
  }
}

// Fades the whole thread in, for when a chat is opened or a new one starts
function fadeInThread() {
  els.messages.classList.remove("fade-in");
  void els.messages.offsetWidth;  // restarts the animation
  els.messages.classList.add("fade-in");
}

function showEmpty() {
  const box = el("div", "empty");
  box.append(
    el("h2", null, "Start a conversation"),
    el("p", null, "Tick “Use web tool” to make the answer come from a web search. Otherwise the assistant decides when it needs one."),
  );
  els.messages.replaceChildren(box);
  fadeInThread();
}

function addUser(text, live = false) {
  const row = el("div", live ? "msg user enter" : "msg user");
  row.append(el("div", "bubble", text));
  els.messages.append(row);
  scrollToEnd();
}

function addAssistant(text) {
  const row = el("div", "msg ai");
  row.append(renderMarkdown(text));
  els.messages.append(row);
  return row;
}

// The assistant row while a reply is being worked on: a collapsible list of steps, then the answer
function addPending() {
  const row = el("div", "msg ai enter");
  const steps = el("div", "steps open");
  const toggle = el("button", "steps-toggle", "Working");
  toggle.type = "button";
  toggle.setAttribute("aria-expanded", "true");
  toggle.addEventListener("click", () => {
    toggle.setAttribute("aria-expanded", String(steps.classList.toggle("open")));
  });
  const inner = el("div", "steps-inner");
  const body = el("div", "steps-body");
  body.append(inner);
  steps.append(toggle, body);
  row.append(steps);
  els.messages.append(row);
  scrollToEnd();
  return { row, steps, toggle, inner, count: 0 };
}

function addStep(pending, text) {
  pending.inner.querySelectorAll(".step.live").forEach((s) => s.classList.remove("live"));
  pending.inner.append(el("div", "step live", text));
  pending.count += 1;
  scrollToEnd();
}

// Closes the steps list (animated) and labels it with the number of steps; safe to call twice
function collapseSteps(pending) {
  pending.inner.querySelectorAll(".step.live").forEach((s) => s.classList.remove("live"));
  if (pending.count === 0) {
    pending.steps.remove();
    return;
  }
  pending.toggle.textContent = pending.count === 1 ? "1 step" : `${pending.count} steps`;
  pending.steps.classList.remove("open");
  pending.toggle.setAttribute("aria-expanded", "false");
}

function showError(pending, message) {
  pending.row.append(el("div", "error", message));
  scrollToEnd();
}

function handleEvent(event, pending) {
  if (event.type === "step") {
    addStep(pending, event.text);
  } else if (event.type === "reply") {
    collapseSteps(pending);
    const body = renderMarkdown(event.content);
    body.classList.add("reveal");
    pending.row.append(body);
    revealReply(pending.row);
  } else if (event.type === "error") {
    showError(pending, event.message);
  }
}

// ---------- sidebar ----------
function renderChats() {
  if (state.chats.length === 0) {
    els.list.replaceChildren(el("div", "list-note", "No chats yet"));
    return;
  }
  els.list.replaceChildren(...state.chats.map((chat) => {
    const item = el("button", "chat-item", chat.title);
    item.type = "button";
    item.title = chat.title;
    if (chat.chat_id === state.chatId) item.setAttribute("aria-current", "true");
    item.addEventListener("click", () => openChat(chat.chat_id));
    return item;
  }));
}

async function refreshChats() {
  try {
    state.chats = await (await api("/api/chats")).json();
    renderChats();
  } catch (err) {
    els.list.replaceChildren(el("div", "list-note", `Could not load chats: ${err.message}`));
  }
  const current = state.chats.find((chat) => chat.chat_id === state.chatId);
  els.title.textContent = current ? current.title : "New chat";
}

function closeNav() { els.app.classList.remove("nav-open"); }

// Grey placeholder bars in the shape of a chat, shown while the messages are being fetched
const LOADING_SHAPE = [  // widths of the placeholder bars: a user message, then the lines of the reply
  { user: "38%", reply: ["94%", "82%", "58%"] },
  { user: "54%", reply: ["90%", "96%", "76%", "42%"] },
  { user: "32%", reply: ["88%", "68%"] },
  { user: "47%", reply: ["95%", "84%", "92%", "50%"] },
  { user: "40%", reply: ["92%", "74%", "56%"] },
];

function showLoading() {
  const box = el("div", "loading");
  box.setAttribute("role", "status");
  box.setAttribute("aria-label", "Loading chat");
  let index = 0;
  const bar = (className, width) => {
    const node = el("div", `skeleton ${className}`);
    node.style.width = width;
    node.style.animationDelay = `-${(index++ % 8) * 0.18}s`;  // out of step, so the pulse moves down the page
    return node;
  };
  LOADING_SHAPE.forEach(({ user, reply }) => {
    const lines = el("div", "sk-reply");
    lines.append(...reply.map((width) => bar("sk-line", width)));
    box.append(bar("sk-user", user), lines);
  });
  els.messages.replaceChildren(box);
}

async function openChat(chatId) {
  if (state.busy) return;
  closeNav();
  // Select the chat right away, so the click is acknowledged before the messages arrive
  state.chatId = chatId;
  updateSummaryButton();
  renderChats();
  const current = state.chats.find((chat) => chat.chat_id === chatId);
  els.title.textContent = current ? current.title : "Chat";
  setBusy(true);  // no typing or switching chats until this one is loaded
  els.thread.setAttribute("aria-busy", "true");
  showLoading();
  try {
    const messages = await (await api(`/api/chats/${chatId}/messages`)).json();
    els.messages.replaceChildren();
    if (messages.length === 0) showEmpty();
    messages.forEach((m) => (m.role === "user" ? addUser(m.content) : addAssistant(m.content)));
    fadeInThread();
    scrollToEnd({ smooth: false });
  } catch (err) {
    els.messages.replaceChildren(el("div", "error", `Could not open this chat: ${err.message}`));
  } finally {
    els.thread.removeAttribute("aria-busy");
    setBusy(false);
  }
}

function startNewChat() {
  if (state.busy) return;
  closeNav();
  state.chatId = null;
  updateSummaryButton();
  els.title.textContent = "New chat";
  showEmpty();
  renderChats();
  els.input.focus();
}

// ---------- sending ----------
function setBusy(busy) {
  state.busy = busy;
  els.send.disabled = busy;
  els.input.disabled = busy;
}

function autosize() {
  // Measure the new height, then let the CSS transition move the box from the old height to it
  const before = els.input.offsetHeight;
  els.input.style.height = "auto";
  const after = Math.min(els.input.scrollHeight, 200);
  els.input.style.height = `${before}px`;
  els.input.style.overflowY = after >= 200 ? "auto" : "hidden";
  requestAnimationFrame(() => { els.input.style.height = `${after}px`; });
}

async function streamTurn(chatId, content, pending) {
  const res = await api(`/api/chats/${chatId}/messages`, {
    method: "POST",
    body: JSON.stringify({ content, use_web: els.useWeb.checked }),
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {  // one JSON event per line
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) handleEvent(JSON.parse(line), pending);
    }
  }
}

async function send() {
  const content = els.input.value.trim();
  if (!content || state.busy) return;
  setBusy(true);
  if (els.messages.querySelector(".empty")) els.messages.replaceChildren();
  addUser(content, true);
  els.input.value = "";
  autosize();
  const pending = addPending();
  try {
    if (!state.chatId) {  // a new chat is created on its first message, so empty chats never pile up
      state.chatId = (await (await api("/api/chats", { method: "POST" })).json()).chat_id;
      updateSummaryButton();
    }
    await streamTurn(state.chatId, content, pending);
  } catch (err) {
    showError(pending, `Could not get a reply: ${err.message}`);
  } finally {
    collapseSteps(pending);
    setBusy(false);
    await refreshChats();
    els.input.focus();
  }
}

// ---------- chat summary ----------
function updateSummaryButton() { els.summaryBtn.hidden = !state.chatId; }

async function openSummary() {
  const chatId = state.chatId;
  if (!chatId) return;
  document.body.classList.add("summary-open");
  els.app.inert = true;  // the page behind the panel can't be reached with the keyboard
  els.summaryPanel.inert = false;
  els.summaryClose.focus();
  els.summaryBody.replaceChildren(el("p", "drawer-note", "Loading..."));
  try {
    const data = await (await api(`/api/chats/${chatId}/summary`)).json();
    if (!data.summary) {
      els.summaryBody.replaceChildren(
        el("p", "drawer-note", "No summary yet. One is written once the chat reaches about 12 messages."),
      );
      return;
    }
    els.summaryBody.replaceChildren(
      el("p", "drawer-meta", `Covers the oldest ${data.messages_summarised} messages of this chat.`),
      renderMarkdown(data.summary),
    );
  } catch (err) {
    els.summaryBody.replaceChildren(el("div", "error", `Could not load the summary: ${err.message}`));
  }
}

function closeSummary() {
  document.body.classList.remove("summary-open");
  els.summaryPanel.inert = true;
  els.app.inert = false;
  els.summaryBtn.focus();
}

// ---------- wiring ----------
els.composer.addEventListener("submit", (event) => { event.preventDefault(); send(); });
els.input.addEventListener("input", autosize);
els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    send();
  }
});
els.useWeb.addEventListener("change", savePreference);
els.newChat.addEventListener("click", startNewChat);
els.menu.addEventListener("click", () => els.app.classList.toggle("nav-open"));
els.scrim.addEventListener("click", closeNav);
els.summaryBtn.addEventListener("click", openSummary);
els.summaryClose.addEventListener("click", closeSummary);
els.summaryScrim.addEventListener("click", closeSummary);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && document.body.classList.contains("summary-open")) closeSummary();
});

loadPreference();
showEmpty();
refreshChats();
els.input.focus();
