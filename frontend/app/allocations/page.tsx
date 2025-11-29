'use client'
import { useState, useEffect } from 'react'
import DashboardLayout from '@/components/DashboardLayout'
import { ArrowRight, Check, X, AlertOctagon, TrendingUp, BrainCircuit, ShieldAlert, ThumbsUp, AlertTriangle } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function AllocationsPage() {
    const [plan, setPlan] = useState<any>(null)
    const [loading, setLoading] = useState(false)
    const [critiqueData, setCritiqueData] = useState<any>(null)

    const generatePlan = async () => {
        setLoading(true)
        try {
            const res = await fetch('http://localhost:8000/api/allocation/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ date: new Date().toISOString().split('T')[0] })
            })
            const data = await res.json()
            setPlan(data)

            // Parse critique if it's a string
            try {
                if (typeof data.critique === 'string') {
                    // Sometimes Gemini returns markdown code blocks ```json ... ```
                    const cleanJson = data.critique.replace(/```json/g, '').replace(/```/g, '')
                    setCritiqueData(JSON.parse(cleanJson))
                } else {
                    setCritiqueData(data.critique)
                }
            } catch (e) {
                console.error("Failed to parse critique JSON", e)
                setCritiqueData(null)
            }

        } catch (e) {
            console.error(e)
        }
        setLoading(false)
    }

    // Mock trend data
    const trendData = [
        { day: 'Mon', inflow: 120, capacity: 100 },
        { day: 'Tue', inflow: 132, capacity: 100 },
        { day: 'Wed', inflow: 101, capacity: 100 },
        { day: 'Thu', inflow: 134, capacity: 100 },
        { day: 'Fri', inflow: 190, capacity: 100 },
        { day: 'Sat', inflow: 230, capacity: 100 },
        { day: 'Sun', inflow: 210, capacity: 100 },
    ]

    return (
        <DashboardLayout>
            <div className="space-y-8">
                <div className="flex justify-between items-center">
                    <div>
                        <h1 className="text-3xl font-bold text-foreground">Resource Allocation</h1>
                        <p className="text-muted-foreground">AI-driven patient redistribution and resource optimization</p>
                    </div>
                    <button
                        onClick={generatePlan}
                        disabled={loading}
                        className="flex items-center gap-2 bg-primary text-primary-foreground px-6 py-3 rounded-lg hover:bg-primary/90 disabled:opacity-50"
                    >
                        {loading ? <BrainCircuit className="animate-spin" /> : <BrainCircuit />}
                        {loading ? 'Generating Plan...' : 'Generate New Plan'}
                    </button>
                </div>

                {plan && (
                    <div className="grid gap-8 lg:grid-cols-3">
                        {/* Main Plan Column */}
                        <div className="lg:col-span-2 space-y-6">
                            {/* Allocations List */}
                            <div className="space-y-4">
                                {plan.allocations?.map((alloc: any, i: number) => (
                                    <div key={i} className="p-6 rounded-xl border border-border bg-card shadow-sm flex flex-col md:flex-row justify-between items-center gap-4">
                                        <div className="flex items-center gap-4 flex-1">
                                            <div className="p-3 bg-red-100 text-red-700 rounded-full font-bold">
                                                {alloc.source_hospital_id}
                                            </div>
                                            <ArrowRight className="text-muted-foreground" />
                                            <div className="p-3 bg-green-100 text-green-700 rounded-full font-bold">
                                                {alloc.target_hospital_id}
                                            </div>
                                            <div>
                                                <h3 className="font-bold text-lg">Move {alloc.patient_count} Patients</h3>
                                                <p className="text-sm text-muted-foreground">{alloc.reason}</p>
                                            </div>
                                        </div>
                                        <div className="flex gap-2">
                                            <button className="p-2 text-green-600 hover:bg-green-50 rounded-full border border-green-200">
                                                <Check size={20} />
                                            </button>
                                            <button className="p-2 text-red-600 hover:bg-red-50 rounded-full border border-red-200">
                                                <X size={20} />
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>

                            {/* AI Critique - Formatted */}
                            <div className="p-6 rounded-xl border border-orange-200 bg-orange-50 dark:bg-orange-900/10 dark:border-orange-800">
                                <div className="flex items-center justify-between mb-6">
                                    <div className="flex items-center gap-2 text-orange-800 dark:text-orange-400">
                                        <AlertOctagon size={24} />
                                        <h3 className="font-bold text-xl">AI Critique & Analysis</h3>
                                    </div>
                                    {critiqueData?.risk_score && (
                                        <div className={`px-4 py-1 rounded-full font-bold text-sm ${critiqueData.risk_score > 7 ? 'bg-red-100 text-red-700' :
                                                critiqueData.risk_score > 4 ? 'bg-yellow-100 text-yellow-700' :
                                                    'bg-green-100 text-green-700'
                                            }`}>
                                            Risk Score: {critiqueData.risk_score}/10
                                        </div>
                                    )}
                                </div>

                                {critiqueData ? (
                                    <div className="space-y-6">
                                        {/* Concerns */}
                                        <div>
                                            <h4 className="flex items-center gap-2 font-semibold text-red-700 mb-3">
                                                <AlertTriangle size={18} /> Key Concerns
                                            </h4>
                                            <ul className="space-y-2">
                                                {critiqueData.concerns?.map((concern: string, i: number) => (
                                                    <li key={i} className="flex gap-2 text-sm text-foreground/80">
                                                        <span className="text-red-500">•</span> {concern}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>

                                        {/* Recommendations */}
                                        <div>
                                            <h4 className="flex items-center gap-2 font-semibold text-green-700 mb-3">
                                                <ThumbsUp size={18} /> Recommendations
                                            </h4>
                                            <ul className="space-y-2">
                                                {critiqueData.recommendations?.map((rec: string, i: number) => (
                                                    <li key={i} className="flex gap-2 text-sm text-foreground/80">
                                                        <span className="text-green-500">•</span> {rec}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>

                                        {/* Sanity Check */}
                                        <div className="flex items-center gap-2 pt-4 border-t border-orange-200/50">
                                            <span className="font-semibold text-sm">Sanity Check:</span>
                                            <span className={`font-bold ${critiqueData.sanity_check === 'PASS' ? 'text-green-600' : 'text-red-600'}`}>
                                                {critiqueData.sanity_check}
                                            </span>
                                        </div>
                                    </div>
                                ) : (
                                    <p className="whitespace-pre-wrap text-foreground">{plan.critique}</p>
                                )}
                            </div>
                        </div>

                        {/* Sidebar Stats */}
                        <div className="space-y-6">
                            <div className="p-6 rounded-xl border border-border bg-card shadow-sm">
                                <h3 className="font-bold text-lg mb-4 flex items-center gap-2">
                                    <TrendingUp size={20} /> Inflow Trend
                                </h3>
                                <div className="h-[200px] w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={trendData}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                                            <XAxis dataKey="day" fontSize={12} stroke="#888" />
                                            <YAxis fontSize={12} stroke="#888" />
                                            <Tooltip />
                                            <Line type="monotone" dataKey="inflow" stroke="#3b82f6" strokeWidth={2} dot={false} />
                                            <Line type="monotone" dataKey="capacity" stroke="#10b981" strokeDasharray="5 5" dot={false} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                                <p className="text-xs text-muted-foreground mt-2 text-center">
                                    City-wide inflow vs capacity (7 days)
                                </p>
                            </div>

                            <div className="p-6 rounded-xl border border-border bg-card shadow-sm">
                                <h3 className="font-bold text-lg mb-4">Summary</h3>
                                <div className="space-y-3 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-muted-foreground">Total Moves</span>
                                        <span className="font-bold">{plan.allocations?.reduce((acc: any, curr: any) => acc + curr.patient_count, 0) || 0}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-muted-foreground">Hospitals Involved</span>
                                        <span className="font-bold">{new Set(plan.allocations?.flatMap((a: any) => [a.source_hospital_id, a.target_hospital_id]) || []).size}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-muted-foreground">Risk Level</span>
                                        <span className="text-orange-600 font-bold">Medium</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {!plan && !loading && (
                    <div className="text-center py-20 border-2 border-dashed border-border rounded-xl">
                        <BrainCircuit className="h-16 w-16 mx-auto text-muted-foreground mb-4 opacity-50" />
                        <h3 className="text-xl font-bold text-foreground">No Active Allocation Plan</h3>
                        <p className="text-muted-foreground">Generate a new plan to see AI recommendations.</p>
                    </div>
                )}
            </div>
        </DashboardLayout>
    )
}
