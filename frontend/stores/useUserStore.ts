import { create } from 'zustand'
import { supabase } from '@/utils/supabase'

interface UserState {
    user: any | null
    role: string | null
    loading: boolean
    checkUser: () => Promise<void>
    logout: () => Promise<void>
}

export const useUserStore = create<UserState>((set) => ({
    user: null,
    role: null,
    loading: true,
    checkUser: async () => {
        const { data: { user } } = await supabase.auth.getUser()
        if (user) {
            // Fetch role from users table
            const { data } = await supabase
                .from('users')
                .select('role')
                .eq('id', user.id)
                .single()

            set({ user, role: data?.role || 'patient', loading: false })
        } else {
            set({ user: null, role: null, loading: false })
        }
    },
    logout: async () => {
        await supabase.auth.signOut()
        set({ user: null, role: null })
    }
}))
