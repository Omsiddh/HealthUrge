'use client'
import Link from 'next/link'
import { Activity, Shield, Users, ArrowRight, BrainCircuit } from 'lucide-react'

export default function LandingPage() {
    return (
        <div className="min-h-screen bg-background">
            {/* Navbar */}
            <nav className="border-b border-border bg-card/50 backdrop-blur-xl fixed w-full z-50">
                <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
                    <div className="flex items-center gap-2">
                        <Activity className="h-6 w-6 text-primary" />
                        <span className="text-xl font-bold text-foreground">MumbaiHacks</span>
                    </div>
                    <div className="flex items-center gap-4">
                        <Link href="/login" className="text-sm font-medium text-muted-foreground hover:text-foreground">
                            Login
                        </Link>
                        <Link
                            href="/login"
                            className="rounded-full bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90"
                        >
                            Get Started
                        </Link>
                    </div>
                </div>
            </nav>

            {/* Hero */}
            <section className="pt-32 pb-20 px-6">
                <div className="mx-auto max-w-7xl text-center">
                    <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-3 py-1 text-sm font-medium text-primary mb-8">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                        </span>
                        Live Healthcare Surveillance
                    </div>
                    <h1 className="text-5xl font-bold tracking-tight text-foreground sm:text-7xl mb-6">
                        AI-Driven <span className="text-primary">Surge Prediction</span> <br />
                        for Resilient Cities.
                    </h1>
                    <p className="mx-auto max-w-2xl text-lg text-muted-foreground mb-10">
                        Optimize hospital resources, predict patient inflow, and provide real-time advisory using advanced Graph Neural Networks and Gemini AI.
                    </p>
                    <div className="flex justify-center gap-4">
                        <Link
                            href="/login"
                            className="flex items-center gap-2 rounded-lg bg-primary px-8 py-4 text-lg font-semibold text-primary-foreground hover:bg-primary/90 transition-all"
                        >
                            Access Dashboard <ArrowRight size={20} />
                        </Link>
                        <Link
                            href="/advisory"
                            className="flex items-center gap-2 rounded-lg border border-border bg-card px-8 py-4 text-lg font-semibold text-foreground hover:bg-muted transition-all"
                        >
                            Check Symptoms
                        </Link>
                    </div>
                </div>
            </section>

            {/* Features */}
            <section className="py-20 bg-muted/50">
                <div className="mx-auto max-w-7xl px-6">
                    <div className="grid gap-8 md:grid-cols-3">
                        <div className="rounded-2xl bg-card p-8 shadow-sm border border-border">
                            <div className="mb-4 inline-flex rounded-lg bg-blue-100 p-3 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400">
                                <BrainCircuit size={24} />
                            </div>
                            <h3 className="mb-2 text-xl font-bold text-foreground">Predictive AI</h3>
                            <p className="text-muted-foreground">
                                NeuralProphet and GNN models forecast patient surges 7 days in advance with 94% accuracy.
                            </p>
                        </div>
                        <div className="rounded-2xl bg-card p-8 shadow-sm border border-border">
                            <div className="mb-4 inline-flex rounded-lg bg-emerald-100 p-3 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400">
                                <Shield size={24} />
                            </div>
                            <h3 className="mb-2 text-xl font-bold text-foreground">Smart Allocation</h3>
                            <p className="text-muted-foreground">
                                Automated resource redistribution plans critiqued by Gemini agents to prevent hospital overload.
                            </p>
                        </div>
                        <div className="rounded-2xl bg-card p-8 shadow-sm border border-border">
                            <div className="mb-4 inline-flex rounded-lg bg-purple-100 p-3 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400">
                                <Users size={24} />
                            </div>
                            <h3 className="mb-2 text-xl font-bold text-foreground">Citizen Advisory</h3>
                            <p className="text-muted-foreground">
                                Personalized triage and hospital recommendations based on real-time capacity and location.
                            </p>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    )
}
