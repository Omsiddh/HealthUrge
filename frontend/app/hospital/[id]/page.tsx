'use client'
import { useState, useEffect } from 'react'
import DashboardLayout from '@/components/DashboardLayout'
import { useParams } from 'next/navigation'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Activity, Users, AlertTriangle, Zap } from 'lucide-react'

export default function HospitalDetailPage() {
    const params = useParams()
    const id = params?.id
    const [forecast, setForecast] = useState<any>(null)
    const [surgeFactor, setSurgeFactor] = useState(1.0)

    useEffect(() => {
        if (id) fetchForecast()
    }, [id])

    const fetchForecast = async () => {
        // Fetch Inflow
        const resInflow = await fetch(`http://localhost:8000/api/forecast?hospital_id=${id}&days=7`)
        const dataInflow = await resInflow.json()

        // Fetch Capacity (mocking separate endpoint call logic for now, assuming same structure)
        // Ideally backend should return both or we call twice. 
        // For hackathon speed, let's assume we just visualize inflow vs a static capacity line or fetch capacity too.
        // Let's fetch capacity too.
        // Wait, the backend endpoint /api/forecast returns "predicted_inflow" and "predicted_capacity" if we update it?
        // Currently it returns just one list based on 'type' param? No, the router calls service.get_forecast.
        // Let's check backend router. It returns {dates, predicted_inflow, predicted_capacity}.

        setForecast(dataInflow)
    }

    const handleSurge = async (factor: number) => {
        setSurgeFactor(factor)
        await fetch('http://localhost:8000/api/admin/simulate-surge', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hospital_id: Number(id), factor })
        })
        fetchForecast()
    }

    if (!forecast) return <DashboardLayout><div>Loading...</div></DashboardLayout>

    const chartData = forecast.dates?.map((date: string, i: number) => ({
        date,
        inflow: forecast.predicted_inflow?.[i] || 0,
        capacity: forecast.predicted_capacity?.[i] || 0
    })) || []

    return (
        <DashboardLayout>
            <div className="space-y-8">
                <div className="flex justify-between items-center">
                    <div>
                        <h1 className="text-3xl font-bold text-foreground">Hospital {id} Dashboard</h1>
                        <p className="text-muted-foreground">Real-time monitoring and 7-day forecast</p>
                    </div>
                    <div className="flex gap-2">
                        <button onClick={() => handleSurge(1.2)} className="flex items-center gap-2 px-4 py-2 bg-orange-100 text-orange-700 rounded-lg hover:bg-orange-200">
                            <Zap size={18} /> Simulate +20% Surge
                        </button>
                        <button onClick={() => handleSurge(1.0)} className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200">
                            Reset
                        </button>
                    </div>
                </div>

                {/* KPI Cards */}
                <div className="grid gap-6 md:grid-cols-3">
                    <div className="p-6 rounded-xl border bg-card shadow-sm">
                        <div className="flex justify-between items-start">
                            <div>
                                <p className="text-sm text-muted-foreground">Predicted Inflow (Today)</p>
                                <h3 className="text-2xl font-bold">{Math.round(chartData[0]?.inflow || 0)}</h3>
                            </div>
                            <Users className="text-blue-500" />
                        </div>
                    </div>
                    <div className="p-6 rounded-xl border bg-card shadow-sm">
                        <div className="flex justify-between items-start">
                            <div>
                                <p className="text-sm text-muted-foreground">Capacity Status</p>
                                <h3 className="text-2xl font-bold">
                                    {Math.round((chartData[0]?.inflow / chartData[0]?.capacity) * 100)}%
                                </h3>
                            </div>
                            <Activity className="text-green-500" />
                        </div>
                    </div>
                    <div className="p-6 rounded-xl border bg-card shadow-sm">
                        <div className="flex justify-between items-start">
                            <div>
                                <p className="text-sm text-muted-foreground">Surge Status</p>
                                <h3 className={`text-2xl font-bold ${surgeFactor > 1 ? 'text-red-500' : 'text-foreground'}`}>
                                    {surgeFactor > 1 ? 'Active' : 'Normal'}
                                </h3>
                            </div>
                            <AlertTriangle className={surgeFactor > 1 ? 'text-red-500' : 'text-gray-300'} />
                        </div>
                    </div>
                </div>

                {/* Chart */}
                <div className="p-6 rounded-xl border bg-card shadow-sm">
                    <h3 className="text-lg font-semibold mb-6">7-Day Forecast: Inflow vs Capacity</h3>
                    <div className="h-[400px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                                <XAxis dataKey="date" stroke="#888" fontSize={12} />
                                <YAxis stroke="#888" fontSize={12} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: 'white', borderRadius: '8px', border: '1px solid #eee' }}
                                />
                                <Legend />
                                <Line type="monotone" dataKey="inflow" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4 }} name="Predicted Inflow" />
                                <Line type="monotone" dataKey="capacity" stroke="#10b981" strokeWidth={2} strokeDasharray="5 5" name="Total Capacity" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </DashboardLayout>
    )
}
