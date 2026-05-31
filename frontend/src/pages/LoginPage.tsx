import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { adminLogin, login } from "../api/auth";
import AuthCard from "../components/AuthCard";
import { useAuth } from "../context/AuthContext";

type LoginMode = "user" | "admin";

export default function LoginPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { setAdminToken, setToken } = useAuth();
  const [mode, setMode] = useState<LoginMode>("user");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const searchMode = new URLSearchParams(location.search).get("mode");
    if (searchMode === "admin") {
      setMode("admin");
      setEmail("admin@system.com");
      setPassword("Admin@123");
    }
  }, [location.search]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setLoading(true);
    try {
      if (mode === "admin") {
        const response = await adminLogin(email, password);
        if (!response.success || !response.token) {
          setError(response.message || "Invalid admin credentials");
          return;
        }
        setAdminToken(response.token);
        navigate("/admin/dashboard");
        return;
      }

      const response = await login(email, password);
      setToken(response.access_token);
      navigate("/");
    } catch {
      setError(mode === "admin" ? "Unable to log in as admin." : "Unable to log in. Please check your credentials.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden bg-[#081A35] px-6 py-6">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(34,211,238,0.08),transparent_28%),radial-gradient(circle_at_bottom,rgba(59,130,246,0.08),transparent_30%)]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-cyan-300/5 to-transparent" />
      <div className="relative z-10 flex w-full max-w-[720px] flex-col items-center">
        <div className="mb-10 text-center">
          <h1 className="text-6xl font-semibold leading-normal text-slate-100">
            Heuri<span className="text-cyan-300">Sense</span>
          </h1>
          <p className="mt-3 text-xs font-semibold uppercase leading-normal tracking-[0.28em] text-slate-400">
            AI Feedback Analytics Platform
          </p>
        </div>
        <AuthCard
          mode="login"
          badgeLabel={mode === "admin" ? "Admin Access" : "AI Feedback Collector"}
          title={mode === "admin" ? "Admin sign in" : "Welcome back"}
          subtitle={
            mode === "admin"
              ? "Use the separate admin access flow to review analytics, users, and collected insights."
              : "Sign in to review feedback and continue collecting structured input."
          }
          submitLabel={mode === "admin" ? "Login to Admin" : "Login to Dashboard"}
          alternateText={mode === "admin" ? "Need user access instead?" : "Need an account?"}
          alternateLabel={mode === "admin" ? "Switch to user login" : "Create one"}
          alternateLink={mode === "admin" ? "/login" : "/signup"}
          onSubmit={handleSubmit}
          email={email}
          password={password}
          setEmail={setEmail}
          setPassword={setPassword}
          error={error}
          loading={loading}
          topTabs={[
            {
              label: "Login",
              active: true,
            },
            {
              label: "Sign up",
              href: "/signup",
            },
          ]}
          auxiliaryAction={{
            label: "Forgot?",
          }}
          adminAction={
            mode === "user"
              ? {
                  label: "Login as Admin",
                  onClick: () => {
                    setMode("admin");
                    setEmail("admin@system.com");
                    setPassword("Admin@123");
                    navigate("/login?mode=admin", { replace: true });
                  },
                }
              : undefined
          }
        />

        <div className="mt-10 flex flex-wrap items-center justify-center gap-8 text-sm leading-normal text-slate-300">
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-cyan-300 shadow-[0_0_14px_rgba(34,211,238,0.7)]" />
            Trusted by Enterprise
          </div>
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-cyan-300 shadow-[0_0_14px_rgba(34,211,238,0.7)]" />
            SOC2 Compliant
          </div>
        </div>

        <div className="mt-24 flex flex-wrap items-center justify-center gap-8 text-xs uppercase leading-normal tracking-[0.22em] text-slate-500">
          <span>Privacy Policy</span>
          <span>Terms of Service</span>
          <span>Security</span>
        </div>
        <p className="mt-5 text-center text-xs uppercase leading-normal tracking-[0.16em] text-slate-600">
          © 2024 Luminescent Insight. All rights reserved.
        </p>
      </div>
    </main>
  );
}
