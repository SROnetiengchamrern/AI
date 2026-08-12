"""Firebase Auth setup for Khmer AI Video Tools.

## Social login (Google / GitHub / Apple)

1. Open: https://console.firebase.google.com/project/ai-khmer-generate/authentication/providers
2. Enable each provider you want:
   - **Google** → Enable → Save (easiest)
   - **GitHub** → Enable → paste Client ID + Client Secret from GitHub OAuth App
   - **Apple** → Enable → follow Apple Developer setup
   - **Email/Password** → Enable (for email form)
3. Authorized domains must include `localhost`
4. Refresh login/register and use **Continue with …**

### GitHub OAuth app (required for GitHub button)
1. https://github.com/settings/developers → New OAuth App
2. Homepage: http://127.0.0.1:7860
3. Callback URL (copy from Firebase GitHub provider screen), usually:
   https://ai-khmer-generate.firebaseapp.com/__/auth/handler
4. Paste Client ID + Secret into Firebase GitHub provider → Save

### SAML SSO (optional)
1. Enable SAML provider in Firebase Console
2. Add to firebase_config.json:
   "samlProviderId": "saml.your-provider-id"

### Passkey (optional)
Needs Firebase Passkeys support + a compatible browser.
If not available, the button shows a clear message — use Google or email.

## App URLs

  http://127.0.0.1:7860/                  → login first
  http://127.0.0.1:7860/auth/login.html
  http://127.0.0.1:7860/auth/register.html
  http://127.0.0.1:7860/app               → tools after login

Optional: FIREBASE_REQUIRE_LOGIN=0 to allow guests into /app.
"""
