'use client'
import { useState, useEffect } from 'react'
import DashboardLayout from '@/components/DashboardLayout'
import { Plus, Save, Activity, AlertTriangle } from 'lucide-react'
import dynamic from 'next/dynamic'

const CityMap = dynamic(() => import('@/components/CityMap'), {
    ssr: false,
    loading: () => <div className="h-[400px] w-full animate-pulse bg-slate-100 rounded-xl"></div>
})

export default function AdminPage() {
    const [hospitals, setHospitals] = useState<any[]>([])
    const [showAddModal, setShowAddModal] = useState(false)
    const [newHospital, setNewHospital] = useState({ name: '', lat: 19.0, lon: 72.8, beds_capacity: 100 })

    useEffect(() => {
        fetchHospitals()
    }, [])

    const fetchHospitals = async () => {
        const res = await fetch('http://localhost:8000/api/admin/hospitals')
        const data = await res.json()
        setHospitals(data)
    }

    const handleAddHospital = async () => {
        await fetch('http://localhost:8000/api/admin/add-hospital', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newHospital)
        })
        setShowAddModal(false)
        fetchHospitals()
    }

    return (
        <DashboardLayout>
            <div className="space-y-8">
                <div className="flex justify-between items-center">
                    <div>
                        <h1 className="text-3xl font-bold text-foreground">City Command Center</h1>
                        <p className="text-muted-foreground">Monitor and manage city-wide healthcare resources</p>
                    </div>
                    <button
                        onClick={() => setShowAddModal(true)}
                        className="flex items-center gap-2 bg-primary text-primary-foreground px-4 py-2 rounded-lg hover:bg-primary/90"
                    >
                        <Plus size={20} /> Add Hospital
                    </button>
                </div>

                {/* Map Section */}
                <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
                    <h3 className="mb-4 text-lg font-semibold text-foreground">Hospital Network Status</h3>
                    <div className="h-[400px] w-full">
                        <CityMap hospitals={hospitals} />
                    </div>
                </div>

                {/* Hospital List */}
                <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                    {hospitals.map(h => (
                        <div key={h.id} className="p-6 rounded-xl border border-border bg-card shadow-sm hover:shadow-md transition-all">
                            <div className="flex justify-between items-start mb-4">
                                <h3 className="font-bold text-lg text-foreground">{h.name}</h3>
                                <span className="bg-blue-100 text-blue-700 text-xs px-2 py-1 rounded-full">ID: {h.id}</span>
                            </div>
                            <div className="space-y-2 text-sm text-muted-foreground">
                                <p>Capacity: {h.beds_capacity} Beds</p>
                                <p>Location: {h.lat}, {h.lon}</p>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Add Hospital Modal */}
                {showAddModal && (
                    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                        <div className="bg-card p-8 rounded-xl w-full max-w-md border border-border">
                            <h2 className="text-2xl font-bold mb-6 text-foreground">Add New Hospital</h2>
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium mb-1">Name</label>
                                    <input
                                        className="w-full p-2 border rounded bg-background text-foreground"
                                        value={newHospital.name}
                                        onChange={e => setNewHospital({ ...newHospital, name: e.target.value })}
                                    />
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm font-medium mb-1">Latitude</label>
                                        <input
                                            type="number"
                                            className="w-full p-2 border rounded bg-background text-foreground"
                                            value={newHospital.lat}
                                            onChange={e => setNewHospital({ ...newHospital, lat: parseFloat(e.target.value) })}
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium mb-1">Longitude</label>
                                        <input
                                            type="number"
                                            className="w-full p-2 border rounded bg-background text-foreground"
                                            value={newHospital.lon}
                                            onChange={e => setNewHospital({ ...newHospital, lon: parseFloat(e.target.value) })}
                                        />
                                    </div>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium mb-1">Beds Capacity</label>
                                    <input
                                        type="number"
                                        className="w-full p-2 border rounded bg-background text-foreground"
                                        value={newHospital.beds_capacity}
                                        onChange={e => setNewHospital({ ...newHospital, beds_capacity: parseInt(e.target.value) })}
                                    />
                                </div>
                                <div className="flex justify-end gap-3 mt-6">
                                    <button
                                        onClick={() => setShowAddModal(false)}
                                        className="px-4 py-2 text-muted-foreground hover:text-foreground"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        onClick={handleAddHospital}
                                        className="px-4 py-2 bg-primary text-primary-foreground rounded hover:bg-primary/90"
                                    >
                                        Save Hospital
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </DashboardLayout>
    )
}
