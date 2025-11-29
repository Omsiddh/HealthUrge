'use client'
import { MapContainer, TileLayer, Marker, Popup, CircleMarker } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'
import { useEffect, useState } from 'react'

// Fix Leaflet icon issue
const icon = L.icon({
    iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
    iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
    shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
})

interface Hospital {
    id: number
    name: string
    lat: number
    lon: number
    stress_score?: number
    stress_level?: string
}

export default function CityMap({ hospitals }: { hospitals: Hospital[] }) {
    const [mounted, setMounted] = useState(false)

    useEffect(() => {
        setMounted(true)
    }, [])

    if (!mounted) return <div className="h-[400px] w-full animate-pulse bg-slate-100 rounded-xl"></div>

    const getColor = (level?: string) => {
        switch (level) {
            case 'high': return '#ef4444'
            case 'moderate': return '#f59e0b'
            case 'low': return '#10b981'
            default: return '#3b82f6'
        }
    }

    return (
        <MapContainer
            center={[19.0760, 72.8777]}
            zoom={11}
            style={{ height: '100%', width: '100%', borderRadius: '0.75rem' }}
        >
            <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {hospitals.map(h => (
                <CircleMarker
                    key={h.id}
                    center={[h.lat, h.lon]}
                    radius={10}
                    pathOptions={{
                        color: getColor(h.stress_level),
                        fillColor: getColor(h.stress_level),
                        fillOpacity: 0.7
                    }}
                >
                    <Popup>
                        <div className="p-2">
                            <h3 className="font-bold">{h.name}</h3>
                            <p className="text-sm">Status: {h.stress_level?.toUpperCase() || 'NORMAL'}</p>
                        </div>
                    </Popup>
                </CircleMarker>
            ))}
        </MapContainer>
    )
}
