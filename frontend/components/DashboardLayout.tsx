'use client'
import { useUserStore } from '@/stores/useUserStore'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { LogOut, LayoutDashboard, Building2, Users, Activity, Settings } from 'lucide-react'

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
    const router = useRouter()
    const logout = useUserStore(state => state.logout)

    const handleLogout = async () => {
        await logout()
        router.push('/login')
    }

    return (
        <div className="flex min-h-screen bg-background">
            {/* Sidebar */}
            <aside className="w-64 border-r border-border bg-card">
                <div className="flex h-16 items-center border-b border-border px-6">
                    <span className="text-xl font-bold text-primary">MumbaiHacks</span>
                </div>
                <nav className="space-y-1 p-4">
                    <Link href="/admin" className="flex items-center gap-3 rounded-lg px-4 py-3 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors">
                        <LayoutDashboard size={20} />
                        Dashboard
                    </Link>
                    <Link href="/allocations" className="flex items-center gap-3 rounded-lg px-4 py-3 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors">
                        <Users size={20} />
                        Allocations
                    </Link>
                    <Link href="/advisory" className="flex items-center gap-3 rounded-lg px-4 py-3 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors">
                        <Activity size={20} />
                        Advisory
                    </Link>
                    <Link href="/settings" className="flex items-center gap-3 rounded-lg px-4 py-3 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors">
                        <Settings size={20} />
                        Settings
                    </Link>
                </nav>
                <div className="absolute bottom-4 w-64 px-4">
                    <button
                        onClick={handleLogout}
                        className="flex w-full items-center gap-3 rounded-lg px-4 py-3 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                    >
                        <LogOut size={20} />
                        Logout
                    </button>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 overflow-y-auto p-8 bg-background">
                {children}
            </main>
        </div>
    )
}
