'use client'
import { useState } from 'react'
import DashboardLayout from '@/components/DashboardLayout'
import { AlertCircle, MapPin, Activity, CheckCircle } from 'lucide-react'

export default function AdvisoryPage() {
    const [form, setForm] = useState({ symptoms: '', lat: 19.076, lon: 72.877, severity: 'Medium' })
    const [advice, setAdvice] = useState<any>(null)
    const [loading, setLoading] = useState(false)

    const handleSubmit = async () => {
        setLoading(true)
        try {
            const res = await fetch('http://localhost:8000/api/advisory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(form)
            })
            const data = await res.json()
            setAdvice(data)
        } catch (e) {
            console.error(e)
        }
        setLoading(false)
    }

    return (
        <DashboardLayout>
            <div className="max-w-3xl mx-auto space-y-8">
                <div>
                    <h1 className="text-3xl font-bold text-foreground">Patient Advisory</h1>
                    <p className="text-muted-foreground">AI-powered triage and hospital recommendation</p>
                </div>

                <div className="grid gap-8 md:grid-cols-2">
                    {/* Form */}
                    <div className="space-y-6 p-6 bg-card border border-border rounded-xl shadow-sm">
                        <div>
                            <label className="block text-sm font-medium mb-2">Symptoms</label>
                            <textarea
                                className="w-full p-3 border rounded-lg bg-background text-foreground min-h-[100px]"
                                placeholder="Describe your symptoms (e.g., high fever, difficulty breathing)..."
                                value={form.symptoms}
                                onChange={e => setForm({ ...form, symptoms: e.target.value })}
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-2">Severity Perception</label>
                            <div className="flex gap-2">
                                {['Low', 'Medium', 'High'].map(level => (
                                    <button
                                        key={level}
                                        onClick={() => setForm({ ...form, severity: level })}
                                        className={`flex-1 py-2 rounded-lg border ${form.severity === level
                                                ? 'bg-primary text-primary-foreground border-primary'
                                                : 'bg-background text-muted-foreground border-input hover:bg-muted'
                                            }`}
                                    >
                                        {level}
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium mb-2">Latitude</label>
                                <input
                                    type="number"
                                    className="w-full p-2 border rounded bg-background text-foreground"
                                    value={form.lat}
                                    onChange={e => setForm({ ...form, lat: parseFloat(e.target.value) })}
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium mb-2">Longitude</label>
                                <input
                                    type="number"
                                    className="w-full p-2 border rounded bg-background text-foreground"
                                    value={form.lon}
                                    onChange={e => setForm({ ...form, lon: parseFloat(e.target.value) })}
                                />
                            </div>
                        </div>

                        <button
                            onClick={handleSubmit}
                            disabled={loading}
                            className="w-full py-3 bg-primary text-primary-foreground font-bold rounded-lg hover:bg-primary/90 disabled:opacity-50"
                        >
                            {loading ? 'Analyzing...' : 'Get Advice'}
                        </button>
                    </div>

                    {/* Result */}
                    <div className="space-y-4">
                        {advice ? (
                            <div className={`p-6 rounded-xl border shadow-sm ${advice.recommendation.includes('Hospital')
                                    ? 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800'
                                    : 'bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800'
                                }`}>
                                <div className="flex items-center gap-3 mb-4">
                                    {advice.recommendation.includes('Hospital') ? (
                                        <AlertCircle className="h-8 w-8 text-red-600 dark:text-red-400" />
                                    ) : (
                                        <CheckCircle className="h-8 w-8 text-green-600 dark:text-green-400" />
                                    )}
                                    <h2 className="text-2xl font-bold text-foreground">{advice.recommendation}</h2>
                                </div>
                                <p className="text-lg text-foreground mb-4">{advice.reason}</p>
                                {advice.suggested_hospital && (
                                    <div className="flex items-center gap-2 text-muted-foreground bg-background/50 p-3 rounded-lg">
                                        <MapPin size={20} />
                                        <span>Suggested Facility: <strong>{advice.suggested_hospital}</strong></span>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="h-full flex items-center justify-center p-8 border-2 border-dashed border-border rounded-xl text-muted-foreground">
                                <div className="text-center">
                                    <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                                    <p>Enter your symptoms to get AI-powered medical advice.</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </DashboardLayout>
    )
}
