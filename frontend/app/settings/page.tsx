'use client'
import { useState } from 'react'
import DashboardLayout from '@/components/DashboardLayout'
import { Save, Bell, Shield, Calendar } from 'lucide-react'

export default function SettingsPage() {
    const [thresholds, setThresholds] = useState({
        load: 90,
        aqi: 200,
        staff: 10
    })
    const [notifications, setNotifications] = useState({
        email: true,
        push: true,
        sms: false
    })

    const handleSave = () => {
        // Mock save
        alert("Settings saved successfully!")
    }

    return (
        <DashboardLayout>
            <div className="space-y-8 max-w-4xl mx-auto">
                <div>
                    <h1 className="text-3xl font-bold text-foreground">System Settings</h1>
                    <p className="text-muted-foreground">Configure thresholds, alerts, and system preferences</p>
                </div>

                {/* Thresholds */}
                <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
                    <div className="flex items-center gap-3 mb-6">
                        <Shield className="h-6 w-6 text-primary" />
                        <h2 className="text-xl font-semibold text-foreground">Alert Thresholds</h2>
                    </div>
                    <div className="grid gap-6 md:grid-cols-3">
                        <div>
                            <label className="mb-2 block text-sm font-medium text-muted-foreground">
                                Max Load Capacity (%)
                            </label>
                            <input
                                type="number"
                                className="w-full rounded-lg border border-input bg-background p-3 text-foreground"
                                value={thresholds.load}
                                onChange={e => setThresholds({ ...thresholds, load: Number(e.target.value) })}
                            />
                            <p className="mt-1 text-xs text-muted-foreground">Trigger alert when exceeded</p>
                        </div>
                        <div>
                            <label className="mb-2 block text-sm font-medium text-muted-foreground">
                                Critical AQI Level
                            </label>
                            <input
                                type="number"
                                className="w-full rounded-lg border border-input bg-background p-3 text-foreground"
                                value={thresholds.aqi}
                                onChange={e => setThresholds({ ...thresholds, aqi: Number(e.target.value) })}
                            />
                        </div>
                        <div>
                            <label className="mb-2 block text-sm font-medium text-muted-foreground">
                                Min Staff Ratio
                            </label>
                            <input
                                type="number"
                                className="w-full rounded-lg border border-input bg-background p-3 text-foreground"
                                value={thresholds.staff}
                                onChange={e => setThresholds({ ...thresholds, staff: Number(e.target.value) })}
                            />
                        </div>
                    </div>
                </div>

                {/* Notifications */}
                <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
                    <div className="flex items-center gap-3 mb-6">
                        <Bell className="h-6 w-6 text-primary" />
                        <h2 className="text-xl font-semibold text-foreground">Notification Preferences</h2>
                    </div>
                    <div className="space-y-4">
                        <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                            <span className="text-foreground font-medium">Email Alerts</span>
                            <input
                                type="checkbox"
                                className="h-5 w-5 rounded border-gray-300 text-primary focus:ring-primary"
                                checked={notifications.email}
                                onChange={e => setNotifications({ ...notifications, email: e.target.checked })}
                            />
                        </div>
                        <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                            <span className="text-foreground font-medium">Push Notifications</span>
                            <input
                                type="checkbox"
                                className="h-5 w-5 rounded border-gray-300 text-primary focus:ring-primary"
                                checked={notifications.push}
                                onChange={e => setNotifications({ ...notifications, push: e.target.checked })}
                            />
                        </div>
                        <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                            <span className="text-foreground font-medium">SMS Alerts (Critical Only)</span>
                            <input
                                type="checkbox"
                                className="h-5 w-5 rounded border-gray-300 text-primary focus:ring-primary"
                                checked={notifications.sms}
                                onChange={e => setNotifications({ ...notifications, sms: e.target.checked })}
                            />
                        </div>
                    </div>
                </div>

                {/* Event Calendar (Mock) */}
                <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
                    <div className="flex items-center gap-3 mb-6">
                        <Calendar className="h-6 w-6 text-primary" />
                        <h2 className="text-xl font-semibold text-foreground">Event Calendar</h2>
                    </div>
                    <div className="text-sm text-muted-foreground">
                        <p>Manage upcoming festivals and events that may impact patient inflow.</p>
                        <div className="mt-4 p-4 bg-muted rounded-lg border border-border">
                            <div className="flex justify-between items-center mb-2">
                                <span className="font-bold text-foreground">Ganesh Chaturthi</span>
                                <span className="text-xs bg-primary/20 text-primary px-2 py-1 rounded">High Impact</span>
                            </div>
                            <p className="text-xs">Sept 19 - Sept 29, 2025</p>
                        </div>
                    </div>
                </div>

                <div className="flex justify-end">
                    <button
                        onClick={handleSave}
                        className="flex items-center gap-2 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground hover:bg-primary/90"
                    >
                        <Save size={20} />
                        Save Changes
                    </button>
                </div>
            </div>
        </DashboardLayout>
    )
}
