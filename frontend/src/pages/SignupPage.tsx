import { useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";

import { signup } from "../api/auth";
import AuthCard from "../components/AuthCard";
import { useAuth } from "../context/AuthContext";

export default function SignupPage() {
  const navigate = useNavigate();
  const { setToken } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const response = await signup(email, password);
      setToken(response.access_token);
      navigate("/");
    } catch (err) {
      setError("Unable to create your account. Try a different email or a stronger password.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden bg-[#081A35] px-6 py-10">
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
          mode="signup"
          badgeLabel="AI Feedback Collector"
          title="Create your workspace"
          subtitle="Start collecting conversational AI feedback with a polished, structured review flow."
          submitLabel="Create account"
          alternateText="Already have an account?"
          alternateLabel="Log in"
          alternateLink="/login"
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
              href: "/login",
            },
            {
              label: "Sign up",
              active: true,
            },
          ]}
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
