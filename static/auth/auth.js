/* Shared Firebase Auth helpers for login / register pages. */

const CONFIG_URL = "/auth/config.json";
const AUTH_CDN = "https://www.gstatic.com/firebasejs/11.0.2/firebase-auth.js";

export function qs(id) {
  return document.getElementById(id);
}

export function setMsg(el, text, kind) {
  if (!el) return;
  el.textContent = text || "";
  el.className = "msg" + (kind ? ` ${kind}` : "");
}

export async function loadFirebaseConfig() {
  const res = await fetch(CONFIG_URL, { cache: "no-store" });
  if (!res.ok) {
    throw new Error("Firebase config missing. Copy firebase_config.example.json → firebase_config.json");
  }
  const cfg = await res.json();
  if (!cfg.apiKey || String(cfg.apiKey).includes("YOUR_FIREBASE")) {
    throw new Error("Fill in firebase_config.json with your Firebase web app keys.");
  }
  return cfg;
}

export async function initFirebaseAuth() {
  const [{ initializeApp }, { getAuth }] = await Promise.all([
    import("https://www.gstatic.com/firebasejs/11.0.2/firebase-app.js"),
    import(AUTH_CDN),
  ]);
  const cfg = await loadFirebaseConfig();
  const app = initializeApp(cfg);
  return { auth: getAuth(app), cfg };
}

export function friendlyAuthError(err) {
  const code = (err && err.code) || "";
  const map = {
    "auth/configuration-not-found":
      "Authentication is not set up. Enable Email/Password (and social providers) in Firebase Console → Authentication → Sign-in method.",
    "auth/operation-not-allowed":
      "This sign-in method is disabled. Enable it in Firebase Console → Authentication → Sign-in method.",
    "auth/popup-closed-by-user": "Sign-in popup was closed. Try again.",
    "auth/popup-blocked": "Popup was blocked. Allow popups for this site and try again.",
    "auth/cancelled-popup-request": "Sign-in was cancelled. Try again.",
    "auth/account-exists-with-different-credential":
      "An account already exists with this email using another sign-in method. Log in with that method first.",
    "auth/unauthorized-domain":
      "This domain is not authorized. Add localhost in Firebase Console → Authentication → Settings → Authorized domains.",
    "auth/email-already-in-use": "This email is already registered. Try logging in.",
    "auth/invalid-email": "Please enter a valid email address.",
    "auth/weak-password": "Password should be at least 6 characters.",
    "auth/user-not-found": "No account found for this email.",
    "auth/wrong-password": "Incorrect password.",
    "auth/invalid-credential": "Incorrect email or password.",
    "auth/too-many-requests": "Too many attempts. Wait a moment and try again.",
    "auth/network-request-failed": "Network error. Check your internet connection.",
  };
  return map[code] || (err && err.message) || "Something went wrong.";
}

export function goToApp() {
  const next = new URLSearchParams(window.location.search).get("next") || "/app";
  window.location.href = next;
}

export async function requireAuthRedirect() {
  const { auth, cfg } = await initFirebaseAuth();
  const { onAuthStateChanged } = await import(AUTH_CDN);
  return new Promise((resolve) => {
    onAuthStateChanged(auth, (user) => {
      if (user) goToApp();
      else resolve({ auth, cfg });
    });
  });
}

async function providerFor(kind, cfg) {
  const authMod = await import(AUTH_CDN);
  if (kind === "google") {
    const p = new authMod.GoogleAuthProvider();
    p.setCustomParameters({ prompt: "select_account" });
    return p;
  }
  if (kind === "github") {
    const p = new authMod.GithubAuthProvider();
    p.addScope("read:user");
    p.addScope("user:email");
    return p;
  }
  if (kind === "apple") {
    return new authMod.OAuthProvider("apple.com");
  }
  if (kind === "saml") {
    const providerId = (cfg && cfg.samlProviderId) || "";
    if (!providerId) {
      throw Object.assign(new Error("SAML is not configured yet."), {
        code: "auth/operation-not-allowed",
        message:
          "Add samlProviderId to firebase_config.json and enable SAML in Firebase Console.",
      });
    }
    return new authMod.SAMLAuthProvider(providerId);
  }
  throw Object.assign(new Error("Unknown provider"), { code: "auth/operation-not-allowed" });
}

export async function signInWithSocial(auth, kind, cfg) {
  const authMod = await import(AUTH_CDN);

  if (kind === "passkey") {
    if (typeof authMod.signInWithPasskey !== "function") {
      throw Object.assign(new Error("Passkeys are not available in this browser/SDK."), {
        code: "auth/operation-not-allowed",
        message:
          "Passkey sign-in needs Firebase Passkeys enabled and a supported browser. Use Google or email for now.",
      });
    }
    return authMod.signInWithPasskey(auth);
  }

  const provider = await providerFor(kind, cfg);
  return authMod.signInWithPopup(auth, provider);
}

export function wireSocialButtons(auth, cfg, msgEl) {
  const buttons = document.querySelectorAll("[data-social]");
  buttons.forEach((btn) => {
    btn.addEventListener("click", async () => {
      const kind = btn.getAttribute("data-social");
      if (!auth) {
        setMsg(msgEl, "Firebase is not configured yet.", "error");
        return;
      }
      buttons.forEach((b) => {
        b.disabled = true;
      });
      setMsg(msgEl, `Continue with ${btn.textContent.replace(/\s+/g, " ").trim()}…`);
      try {
        await signInWithSocial(auth, kind, cfg);
        setMsg(msgEl, "Success — opening app…", "ok");
        goToApp();
      } catch (err) {
        if (err && err.code === "auth/popup-closed-by-user") {
          setMsg(msgEl, "");
        } else {
          setMsg(msgEl, friendlyAuthError(err), "error");
        }
        buttons.forEach((b) => {
          b.disabled = false;
        });
      }
    });
  });
}

/** Markup for Continue-with buttons (shared by login + register). */
export const SOCIAL_BUTTONS_HTML = `
<div class="social" id="social-buttons">
  <button type="button" class="social-btn" data-social="google">
    <span class="icon" aria-hidden="true">
      <svg viewBox="0 0 24 24"><path fill="#EA4335" d="M12 10.2v3.6h5.1c-.2 1.2-.9 2.3-1.9 3l3.1 2.4c1.8-1.7 2.9-4.1 2.9-7 0-.7-.1-1.3-.2-1.9H12z"/><path fill="#34A853" d="M6.6 14.3l-.8.6-2.7 2.1C4.8 19.7 8.1 22 12 22c2.7 0 4.9-.9 6.5-2.4l-3.1-2.4c-.9.6-2 .9-3.4.9-2.6 0-4.8-1.7-5.6-4.1z"/><path fill="#4A90E2" d="M3.1 7.1C2.4 8.5 2 10.2 2 12s.4 3.5 1.1 4.9l3.5-2.7C6.3 13.4 6.1 12.7 6.1 12c0-.7.2-1.4.5-2.1L3.1 7.1z"/><path fill="#FBBC05" d="M12 5.9c1.5 0 2.8.5 3.8 1.5l2.8-2.8C16.9 2.9 14.7 2 12 2 8.1 2 4.8 4.3 3.1 7.1l3.5 2.7C7.2 7.6 9.4 5.9 12 5.9z"/></svg>
    </span>
    Continue with Google
  </button>
  <button type="button" class="social-btn" data-social="github">
    <span class="icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="#fff"><path d="M12 2C6.5 2 2 6.6 2 12.3c0 4.5 2.9 8.4 6.9 9.7.5.1.7-.2.7-.5v-1.9c-2.8.6-3.4-1.4-3.4-1.4-.5-1.2-1.1-1.5-1.1-1.5-.9-.6.1-.6.1-.6 1 .1 1.5 1 1.5 1 .9 1.6 2.4 1.1 3 .9.1-.7.4-1.1.6-1.4-2.2-.3-4.6-1.2-4.6-5.1 0-1.1.4-2 1-2.7-.1-.3-.4-1.3.1-2.7 0 0 .8-.3 2.8 1 .8-.2 1.6-.3 2.4-.3s1.6.1 2.4.3c1.9-1.3 2.7-1 2.7-1 .5 1.4.2 2.4.1 2.7.6.7 1 1.6 1 2.7 0 4-2.3 4.8-4.6 5.1.4.3.7.9.7 1.9v2.8c0 .3.2.6.7.5 4-1.3 6.9-5.2 6.9-9.7C22 6.6 17.5 2 12 2z"/></svg>
    </span>
    Continue with GitHub
  </button>
  <button type="button" class="social-btn" data-social="apple">
    <span class="icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="#fff"><path d="M16.7 12.6c0-2.1 1.7-3.1 1.8-3.2-1-1.4-2.5-1.6-3-1.7-1.3-.1-2.5.8-3.1.8-.7 0-1.7-.7-2.8-.7-1.4 0-2.8.9-3.5 2.2-1.5 2.6-.4 6.5 1.1 8.6.7 1 1.6 2.2 2.7 2.1 1.1 0 1.5-.7 2.8-.7s1.6.7 2.8.7c1.2 0 1.9-1 2.6-2 .8-1.2 1.1-2.3 1.1-2.4-.1 0-2.2-.8-2.2-3.7zM14.4 6.5c.6-.7 1-1.7.9-2.7-1 .1-2.1.6-2.7 1.4-.6.7-1.1 1.7-.9 2.7 1 0 2-.6 2.7-1.4z"/></svg>
    </span>
    Continue with Apple
  </button>
  <button type="button" class="social-btn" data-social="saml">
    <span class="icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8"><rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/><circle cx="12" cy="16" r="1.2" fill="#fff" stroke="none"/></svg>
    </span>
    Continue with SAML SSO
  </button>
  <button type="button" class="social-btn" data-social="passkey">
    <span class="icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8"><circle cx="9" cy="8" r="3.2"/><path d="M3.5 19c.6-3 2.8-5 5.5-5s4.9 2 5.5 5"/><circle cx="17.5" cy="12.5" r="2"/><path d="M17.5 14.5v4.2M17.5 18.7h2.2"/></svg>
    </span>
    Continue with Passkey
  </button>
</div>
`;
