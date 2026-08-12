/* Auth status bar on the Gradio home page. */

const REQUIRE_LOGIN = window.__VK_REQUIRE_LOGIN__ === true;

function displayNameFor(user) {
  if (!user) return "Account";
  if (user.displayName && user.displayName.trim()) return user.displayName.trim();
  if (user.email) return user.email.split("@")[0] || user.email;
  return "Account";
}

function setSignedInUI(labelText) {
  const guest = document.getElementById("vk-auth-guest");
  const dropdown = document.getElementById("vk-auth-dropdown");
  const label = document.getElementById("vk-auth-user");
  const menu = document.getElementById("vk-auth-menu");
  const trigger = document.getElementById("vk-auth-trigger");

  if (label) label.textContent = labelText || "Account";
  guest?.classList.remove("is-visible");
  dropdown?.classList.add("is-visible");
  menu?.classList.remove("is-open");
  if (trigger) trigger.setAttribute("aria-expanded", "false");
}

function setGuestUI() {
  const guest = document.getElementById("vk-auth-guest");
  const dropdown = document.getElementById("vk-auth-dropdown");
  const menu = document.getElementById("vk-auth-menu");
  const trigger = document.getElementById("vk-auth-trigger");

  guest?.classList.add("is-visible");
  dropdown?.classList.remove("is-visible");
  menu?.classList.remove("is-open");
  if (trigger) trigger.setAttribute("aria-expanded", "false");
}

function wireDropdown(auth, signOut) {
  const trigger = document.getElementById("vk-auth-trigger");
  const menu = document.getElementById("vk-auth-menu");
  const logoutBtn = document.getElementById("vk-auth-logout");
  if (!trigger || !menu || !logoutBtn) return;

  trigger.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = menu.classList.toggle("is-open");
    trigger.setAttribute("aria-expanded", open ? "true" : "false");
  });

  document.addEventListener("click", () => {
    menu.classList.remove("is-open");
    trigger.setAttribute("aria-expanded", "false");
  });

  menu.addEventListener("click", (e) => e.stopPropagation());

  logoutBtn.addEventListener("click", async () => {
    await signOut(auth);
    location.href = "/auth/login.html?next=/app";
  });
}

async function bootAuthBar() {
  if (!document.getElementById("vk-auth-bar")) return;

  try {
    const res = await fetch("/auth/config.json", { cache: "no-store" });
    if (!res.ok) throw new Error("no config");
    const cfg = await res.json();
    if (!cfg.apiKey || String(cfg.apiKey).includes("YOUR_FIREBASE")) {
      throw new Error("placeholder");
    }

    const [{ initializeApp }, { getAuth, onAuthStateChanged, signOut }] = await Promise.all([
      import("https://www.gstatic.com/firebasejs/11.0.2/firebase-app.js"),
      import("https://www.gstatic.com/firebasejs/11.0.2/firebase-auth.js"),
    ]);

    const auth = getAuth(initializeApp(cfg));
    wireDropdown(auth, signOut);

    onAuthStateChanged(auth, (user) => {
      if (user) {
        setSignedInUI(displayNameFor(user));
      } else {
        setGuestUI();
        if (REQUIRE_LOGIN) {
          location.href = "/auth/login.html?next=" + encodeURIComponent("/app");
        }
      }
    });
  } catch (err) {
    setGuestUI();
  }
}

function startWhenReady() {
  if (document.getElementById("vk-auth-bar")) {
    bootAuthBar();
    return;
  }
  const obs = new MutationObserver(() => {
    if (document.getElementById("vk-auth-bar")) {
      obs.disconnect();
      bootAuthBar();
    }
  });
  obs.observe(document.documentElement, { childList: true, subtree: true });
  setTimeout(() => obs.disconnect(), 15000);
}

startWhenReady();
