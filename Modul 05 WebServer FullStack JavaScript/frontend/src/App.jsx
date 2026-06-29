import React, { useState, useEffect, useRef } from 'react';
import { auth, db } from './firebase';
import { 
  signInWithEmailAndPassword, 
  createUserWithEmailAndPassword, 
  signOut, 
  onAuthStateChanged 
} from 'firebase/auth';
import { 
  collection, 
  doc, 
  getDoc, 
  getDocs, 
  addDoc, 
  setDoc, 
  updateDoc, 
  deleteDoc, 
  query, 
  orderBy, 
  serverTimestamp 
} from 'firebase/firestore';

// ======================== ICON COMPONENTS ========================
const Icon = ({ d, size = 20, className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    {Array.isArray(d) ? d.map((path, i) => <path key={i} d={path} />) : <path d={d} />}
  </svg>
);

const MailIcon = ({ size, className }) => <Icon size={size} className={className} d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z M22 6l-10 7L2 6" />;
const SendIcon = ({ size, className }) => <Icon size={size} className={className} d="M22 2L11 13 M22 2L15 22 8 13 2 9z" />;
const GithubIcon = ({ size, className }) => (
  <svg width={size || 20} height={size || 20} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" />
  </svg>
);
const LinkedinIcon = ({ size, className }) => (
  <svg width={size || 20} height={size || 20} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z" />
    <rect x="2" y="9" width="4" height="12" />
    <circle cx="4" cy="4" r="2" />
  </svg>
);
const ExternalLinkIcon = ({ size, className }) => <Icon size={size} className={className} d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6 M15 3h6v6 M10 14L21 3" />;
const MapPinIcon = ({ size, className }) => (
  <svg width={size || 18} height={size || 18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
    <circle cx="12" cy="10" r="3" />
  </svg>
);
const CpuIcon = ({ size, className }) => (
  <svg width={size || 20} height={size || 20} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <rect x="4" y="4" width="16" height="16" rx="2" />
    <rect x="9" y="9" width="6" height="6" />
    <line x1="9" y1="1" x2="9" y2="4" /><line x1="15" y1="1" x2="15" y2="4" />
    <line x1="9" y1="20" x2="9" y2="23" /><line x1="15" y1="20" x2="15" y2="23" />
    <line x1="20" y1="9" x2="23" y2="9" /><line x1="20" y1="14" x2="23" y2="14" />
    <line x1="1" y1="9" x2="4" y2="9" /><line x1="1" y1="14" x2="4" y2="14" />
  </svg>
);
const TerminalIcon = ({ size, className }) => (
  <svg width={size || 20} height={size || 20} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <polyline points="4 17 10 11 4 5" /><line x1="12" y1="19" x2="20" y2="19" />
  </svg>
);
const BriefcaseIcon = ({ size, className }) => (
  <svg width={size || 20} height={size || 20} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <rect x="2" y="7" width="20" height="14" rx="2" ry="2" />
    <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
  </svg>
);
const AwardIcon = ({ size, className }) => (
  <svg width={size || 20} height={size || 20} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <circle cx="12" cy="8" r="7" />
    <path d="M8.21 13.89L7 23l5-3 5 3-1.21-9.12" />
  </svg>
);
const GraduationCapIcon = ({ size, className }) => (
  <svg width={size || 20} height={size || 20} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M22 10v6M2 10l10-5 10 5-10 5z" />
    <path d="M6 12v5c0 2 2 3 6 3s6-1 6-3v-5" />
  </svg>
);
const ShieldIcon = ({ size, className }) => (
  <svg width={size || 18} height={size || 18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
  </svg>
);
const LogOutIcon = ({ size, className }) => (
  <svg width={size || 16} height={size || 16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
    <polyline points="16 17 21 12 16 7" />
    <line x1="21" y1="12" x2="9" y2="12" />
  </svg>
);
const TrashIcon = ({ size, className }) => (
  <svg width={size || 14} height={size || 14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
  </svg>
);
const EditIcon = ({ size, className }) => (
  <svg width={size || 14} height={size || 14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
    <path d="M18.5 2.5a2.121 2.121 0 1 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
  </svg>
);
const CheckIcon = ({ size, className }) => (
  <svg width={size || 14} height={size || 14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
    <polyline points="22 4 12 14.01 9 11.01" />
  </svg>
);
const SlidersIcon = ({ size, className }) => (
  <svg width={size || 20} height={size || 20} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <line x1="4" y1="21" x2="4" y2="14" /><line x1="4" y1="10" x2="4" y2="3" />
    <line x1="12" y1="21" x2="12" y2="12" /><line x1="12" y1="8" x2="12" y2="3" />
    <line x1="20" y1="21" x2="20" y2="16" /><line x1="20" y1="12" x2="20" y2="3" />
    <line x1="2" y1="14" x2="6" y2="14" /><line x1="10" y1="8" x2="14" y2="8" /><line x1="18" y1="16" x2="22" y2="16" />
  </svg>
);
const ChevronRightIcon = ({ size, className }) => (
  <svg width={size || 16} height={size || 16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <polyline points="9 18 15 12 9 6" />
  </svg>
);
const MenuIcon = ({ size, className }) => (
  <svg width={size || 20} height={size || 20} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="18" x2="21" y2="18" />
  </svg>
);
const XIcon = ({ size, className }) => (
  <svg width={size || 20} height={size || 20} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);
const ZapIcon = ({ size, className }) => (
  <svg width={size || 18} height={size || 18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
  </svg>
);

// ======================== ANIMATED COUNTER ========================
function AnimatedCounter({ target, suffix = '' }) {
  const [count, setCount] = useState(0);
  const ref = useRef(null);
  
  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        let start = 0;
        const duration = 1500;
        const step = target / (duration / 16);
        const timer = setInterval(() => {
          start += step;
          if (start >= target) {
            setCount(target);
            clearInterval(timer);
          } else {
            setCount(Math.floor(start));
          }
        }, 16);
        observer.disconnect();
      }
    });
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [target]);
  
  return <span ref={ref}>{count}{suffix}</span>;
}

// ======================== SKILL BAR COMPONENT ========================
function SkillBar({ label, pct, color = '#14b8a6' }) {
  const [animated, setAnimated] = useState(false);
  const ref = useRef(null);
  
  useEffect(() => {
    const observer = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) { setAnimated(true); observer.disconnect(); }
    }, { threshold: 0.3 });
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  
  return (
    <div ref={ref} className="group">
      <div className="flex justify-between mb-2">
        <span className="text-sm font-medium text-slate-300 group-hover:text-teal-400 transition-colors">{label}</span>
        <span className="text-xs font-mono text-teal-400">{pct}%</span>
      </div>
      <div className="skill-bar">
        <div 
          className="skill-bar-fill transition-all duration-1000 ease-out"
          style={{ width: animated ? `${pct}%` : '0%' }}
        />
      </div>
    </div>
  );
}

// ======================== MAIN APP ========================
function App() {
  const [view, setView] = useState('public');
  const [currentUser, setCurrentUser] = useState(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [activeSection, setActiveSection] = useState('about');

  const [profile, setProfile] = useState({
    name: 'REVITA NADA YUANIKA PUTRI',
    title: 'AUTOMATION & IOT ENGINEERING STUDENT',
    summary: 'Highly motivated Sixth-semester student of Automation Engineering Technology at Diponegoro University with a 3.70/4.00 GPA. Skilled in developing automation systems, integrating hardware and software, and applying microcontroller-based technology.',
    gpa: '3.70 / 4.00',
    location: 'Semarang, ID',
    email: 'revitanada05@gmail.com',
    linkedin: 'www.linkedin.com/in/revitanada',
    cvUrl: 'https://bit.ly/4npMvsR',
    github: 'https://github.com'
  });

  const [experiences, setExperiences] = useState([]);
  const [projects, setProjects] = useState([]);
  const [filteredProjects, setFilteredProjects] = useState([]);
  const [activeCategory, setActiveCategory] = useState('All');
  const [formData, setFormData] = useState({ name: '', email: '', message: '' });
  const [status, setStatus] = useState('');
  const [statusType, setStatusType] = useState('');
  const [activeSkillsTab, setActiveSkillsTab] = useState('technical');
  const [activeTimelineTab, setActiveTimelineTab] = useState('organization');
  const [activeDashboardTab, setActiveDashboardTab] = useState('projects');
  const [authMode, setAuthMode] = useState('login');
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authError, setAuthError] = useState('');
  const [authSuccess, setAuthSuccess] = useState('');
  const [adminMessages, setAdminMessages] = useState([]);
  const [editingProject, setEditingProject] = useState(null);
  const [projectForm, setProjectForm] = useState({ title: '', description: '', tags: '', year: '', category: 'Automation & IoT' });
  const [editingExp, setEditingExp] = useState(null);
  const [expForm, setExpForm] = useState({ type: 'organization', title: '', institution: '', period: '', bullets: '', order: 0 });

  // Scroll tracking
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
      const sections = ['about', 'skills', 'experience', 'projects', 'certifications', 'contact'];
      for (const s of sections.reverse()) {
        const el = document.getElementById(s);
        if (el && window.scrollY >= el.offsetTop - 120) { setActiveSection(s); break; }
      }
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      setCurrentUser(user);
      if (user) loadAdminData();
    });
    return () => unsubscribe();
  }, []);

  useEffect(() => {
    fetchProfile(); fetchExperience(); fetchProjects();
  }, []);

  useEffect(() => {
    if (activeCategory === 'All') setFilteredProjects(projects);
    else setFilteredProjects(projects.filter(p => p.category === activeCategory));
  }, [activeCategory, projects]);

  const fetchProfile = async () => {
    try {
      const snap = await getDoc(doc(db, 'profile', 'default'));
      if (snap.exists()) setProfile(snap.data());
    } catch (err) { console.error(err); }
  };

  const fetchExperience = async () => {
    try {
      const q = query(collection(db, 'experience'), orderBy('order', 'asc'));
      const snap = await getDocs(q);
      setExperiences(snap.docs.map(d => ({ id: d.id, ...d.data() })));
    } catch (err) { console.error(err); }
  };

  const fetchProjects = async () => {
    try {
      const snap = await getDocs(collection(db, 'projects'));
      const projs = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      setProjects(projs); setFilteredProjects(projs);
    } catch (err) { console.error(err); }
  };

  const loadAdminData = async () => {
    try {
      const q = query(collection(db, 'messages'), orderBy('createdAt', 'desc'));
      const snap = await getDocs(q);
      setAdminMessages(snap.docs.map(d => ({ id: d.id, ...d.data() })));
    } catch (err) { console.error(err); }
  };

  const handleAuth = async (e) => {
    e.preventDefault();
    setAuthError(''); setAuthSuccess('');
    try {
      if (authMode === 'login') {
        await signInWithEmailAndPassword(auth, authEmail, authPassword);
        setAuthSuccess('Berhasil masuk!');
        setTimeout(() => { setView('dashboard'); setAuthEmail(''); setAuthPassword(''); setAuthSuccess(''); }, 1000);
      } else {
        await createUserWithEmailAndPassword(auth, authEmail, authPassword);
        setAuthSuccess('Akun admin dibuat!');
        setTimeout(() => { setView('dashboard'); setAuthEmail(''); setAuthPassword(''); setAuthSuccess(''); }, 1000);
      }
    } catch (err) {
      if (err.code === 'auth/email-already-in-use') setAuthError('Email sudah terdaftar.');
      else if (err.code === 'auth/wrong-password' || err.code === 'auth/invalid-credential') setAuthError('Email atau password salah.');
      else setAuthError(err.message);
    }
  };

  const handleLogout = async () => { await signOut(auth); setView('public'); };

  const handleContactSubmit = async (e) => {
    e.preventDefault();
    setStatus('Mengirim...'); setStatusType('info');
    try {
      await addDoc(collection(db, 'messages'), { ...formData, createdAt: serverTimestamp() });
      setStatus('Pesan berhasil terkirim! Terima kasih.'); setStatusType('success');
      setFormData({ name: '', email: '', message: '' });
      if (currentUser) loadAdminData();
    } catch (err) {
      setStatus('Gagal mengirim. Coba lagi.'); setStatusType('error');
    }
  };

  const handleProfileUpdate = async (e) => {
    e.preventDefault();
    try {
      await setDoc(doc(db, 'profile', 'default'), profile, { merge: true });
      alert('Profil diperbarui!'); fetchProfile();
    } catch (err) { alert('Gagal: ' + err.message); }
  };

  const handleProjectSubmit = async (e) => {
    e.preventDefault();
    const payload = { ...projectForm, tags: projectForm.tags.split(',').map(t => t.trim()).filter(Boolean) };
    try {
      if (editingProject) {
        await updateDoc(doc(db, 'projects', editingProject.id), payload);
        setEditingProject(null);
      } else {
        await addDoc(collection(db, 'projects'), { ...payload, createdAt: serverTimestamp() });
      }
      setProjectForm({ title: '', description: '', tags: '', year: '', category: 'Automation & IoT' });
      fetchProjects();
    } catch (err) { alert(err.message); }
  };

  const editProject = (p) => {
    setEditingProject(p);
    setProjectForm({ title: p.title, description: p.description, tags: p.tags ? p.tags.join(', ') : '', year: p.year || '', category: p.category || 'Automation & IoT' });
  };

  const deleteProject = async (id) => {
    if (!window.confirm('Hapus proyek ini?')) return;
    try { await deleteDoc(doc(db, 'projects', id)); fetchProjects(); } catch (err) { alert(err.message); }
  };

  const handleExpSubmit = async (e) => {
    e.preventDefault();
    const payload = { ...expForm, bullets: expForm.bullets.split('\n').map(b => b.trim()).filter(Boolean), order: Number(expForm.order) };
    try {
      if (editingExp) { await updateDoc(doc(db, 'experience', editingExp.id), payload); setEditingExp(null); }
      else await addDoc(collection(db, 'experience'), { ...payload, createdAt: serverTimestamp() });
      setExpForm({ type: 'organization', title: '', institution: '', period: '', bullets: '', order: 0 });
      fetchExperience();
    } catch (err) { alert(err.message); }
  };

  const editExp = (eItem) => {
    setEditingExp(eItem);
    setExpForm({ type: eItem.type, title: eItem.title, institution: eItem.institution, period: eItem.period, bullets: eItem.bullets ? eItem.bullets.join('\n') : '', order: eItem.order || 0 });
  };

  const deleteExp = async (id) => {
    if (!window.confirm('Hapus riwayat ini?')) return;
    try { await deleteDoc(doc(db, 'experience', id)); fetchExperience(); } catch (err) { alert(err.message); }
  };

  const deleteMessage = async (id) => {
    if (!window.confirm('Hapus pesan ini?')) return;
    try { await deleteDoc(doc(db, 'messages', id)); loadAdminData(); } catch (err) { alert(err.message); }
  };

  const categories = ['All', 'Automation & IoT', 'Industrial Control', 'Mechanical', 'Programming'];
  const navLinks = [
    { href: '#about', label: 'Tentang' },
    { href: '#skills', label: 'Keahlian' },
    { href: '#experience', label: 'Pengalaman' },
    { href: '#projects', label: 'Proyek' },
    { href: '#certifications', label: 'Sertifikasi' },
    { href: '#contact', label: 'Kontak' },
  ];

  // ======================== LOGIN VIEW ========================
  if (view === 'login') {
    return (
      <div className="min-h-screen bg-[#020817] text-slate-100 flex items-center justify-center p-6 relative overflow-hidden">
        {/* Background */}
        <div className="bg-grid absolute inset-0" />
        <div className="orb-teal w-[600px] h-[600px] top-[-200px] left-[50%] transform -translate-x-1/2" />
        <div className="orb-blue w-[400px] h-[400px] bottom-[-100px] right-[-100px]" />
        
        <div className="relative z-10 w-full max-w-md">
          {/* Back button */}
          <button onClick={() => setView('public')} className="flex items-center gap-2 text-slate-500 hover:text-teal-400 transition text-sm mb-8 group">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="group-hover:-translate-x-1 transition-transform">
              <line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/>
            </svg>
            Kembali ke Portofolio
          </button>

          {/* Card */}
          <div className="glass-card rounded-2xl p-8" style={{ background: 'rgba(15,23,42,0.7)' }}>
            {/* Header */}
            <div className="flex items-center gap-3 mb-8">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-400 to-emerald-500 flex items-center justify-center">
                <ShieldIcon size={18} className="text-slate-950" />
              </div>
              <div>
                <h1 className="font-bold text-white text-lg">Admin Portal</h1>
                <p className="text-xs text-slate-500 font-mono">SECURE_AUTH_v2.0</p>
              </div>
              <div className="ml-auto flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-teal-400 led-blink" />
                <span className="text-[10px] text-teal-400 font-mono font-bold">ONLINE</span>
              </div>
            </div>

            {/* Toggle */}
            <div className="flex bg-[#020817] p-1 rounded-xl border border-slate-800/60 mb-6">
              {['login', 'register'].map(mode => (
                <button key={mode} onClick={() => setAuthMode(mode)}
                  className={`flex-1 py-2.5 text-xs font-bold rounded-lg transition-all ${authMode === mode ? 'bg-teal-500 text-slate-950 shadow-lg' : 'text-slate-500 hover:text-slate-300'}`}>
                  {mode === 'login' ? 'Masuk' : 'Daftar'}
                </button>
              ))}
            </div>

            <form onSubmit={handleAuth} className="space-y-4">
              <div>
                <label className="block text-[11px] font-bold text-slate-500 uppercase tracking-widest mb-2">Email</label>
                <input type="email" value={authEmail} onChange={e => setAuthEmail(e.target.value)} required placeholder="admin@email.com"
                  className="form-input" />
              </div>
              <div>
                <label className="block text-[11px] font-bold text-slate-500 uppercase tracking-widest mb-2">Password</label>
                <input type="password" value={authPassword} onChange={e => setAuthPassword(e.target.value)} required placeholder="••••••••"
                  className="form-input" />
              </div>
              <button type="submit" className="btn-primary w-full py-3 mt-2 text-sm flex items-center justify-center gap-2 cursor-pointer">
                <span>{authMode === 'login' ? 'Masuk Sekarang' : 'Daftarkan Akun'}</span>
                <ChevronRightIcon size={16} />
              </button>
            </form>

            {authError && <div className="mt-4 p-3 rounded-xl text-xs bg-red-500/10 border border-red-500/20 text-red-400 text-center">{authError}</div>}
            {authSuccess && <div className="mt-4 p-3 rounded-xl text-xs bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-center">{authSuccess}</div>}
          </div>
        </div>
      </div>
    );
  }

  // ======================== DASHBOARD VIEW ========================
  if (view === 'dashboard') {
    return (
      <div className="min-h-screen bg-[#020817] text-slate-100">
        <div className="bg-grid fixed inset-0 pointer-events-none" />

        {/* Admin Nav */}
        <nav className="admin-nav sticky top-0 z-50 px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal-400 to-emerald-500 flex items-center justify-center font-extrabold text-slate-950 text-sm">R</div>
            <div>
              <span className="text-xs font-mono font-bold text-teal-400">ADMIN PANEL</span>
              <span className="text-slate-400 text-xs ml-2">— REVITA N.Y.P</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button onClick={() => setView('public')} className="text-xs font-semibold text-slate-400 hover:text-teal-400 transition px-3 py-2 rounded-lg hover:bg-teal-500/5">
              Lihat Publik
            </button>
            <button onClick={handleLogout} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500 hover:text-white transition text-xs font-bold">
              <LogOutIcon size={14} /> Logout
            </button>
          </div>
        </nav>

        <main className="relative z-10 max-w-5xl mx-auto px-6 py-10">
          <div className="mb-8">
            <div className="section-label">dashboard</div>
            <h1 className="text-3xl font-extrabold text-white">Manajemen Portofolio</h1>
            <p className="text-slate-500 text-sm mt-1">Kelola konten, proyek, dan pesan masuk dari satu panel.</p>
          </div>

          {/* Tabs */}
          <div className="flex flex-wrap gap-1 border-b border-slate-800/60 mb-8">
            {[
              { id: 'projects', label: 'Proyek' },
              { id: 'experience', label: 'Riwayat' },
              { id: 'profile', label: 'Profil' },
              { id: 'messages', label: `Pesan (${adminMessages.length})` }
            ].map(tab => (
              <button key={tab.id} onClick={() => setActiveDashboardTab(tab.id)}
                className={`pb-3 px-5 text-sm font-semibold transition border-b-2 -mb-px ${activeDashboardTab === tab.id ? 'border-teal-500 text-teal-400' : 'border-transparent text-slate-500 hover:text-slate-300'}`}>
                {tab.label}
              </button>
            ))}
          </div>

          {/* PROJECTS TAB */}
          {activeDashboardTab === 'projects' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <div className="lg:col-span-5 admin-card p-6 h-fit">
                <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                  {editingProject ? <><EditIcon size={16} /> Edit Proyek</> : <><span className="text-teal-400">+</span> Tambah Proyek</>}
                </h3>
                <form onSubmit={handleProjectSubmit} className="space-y-3">
                  {[
                    { label: 'Judul Proyek', key: 'title', placeholder: 'Boiler Control System', type: 'text' },
                  ].map(f => (
                    <div key={f.key}>
                      <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">{f.label}</label>
                      <input type={f.type} value={projectForm[f.key]} onChange={e => setProjectForm({...projectForm, [f.key]: e.target.value})} required placeholder={f.placeholder} className="admin-input" />
                    </div>
                  ))}
                  <div>
                    <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Deskripsi</label>
                    <textarea value={projectForm.description} onChange={e => setProjectForm({...projectForm, description: e.target.value})} required rows="3" className="admin-input resize-none" placeholder="Jelaskan proyek..." />
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Kategori</label>
                    <select value={projectForm.category} onChange={e => setProjectForm({...projectForm, category: e.target.value})} className="admin-input">
                      {categories.filter(c => c !== 'All').map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Tahun</label>
                      <input type="text" value={projectForm.year} onChange={e => setProjectForm({...projectForm, year: e.target.value})} placeholder="2026" className="admin-input" />
                    </div>
                    <div>
                      <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Tags (koma)</label>
                      <input type="text" value={projectForm.tags} onChange={e => setProjectForm({...projectForm, tags: e.target.value})} placeholder="ESP32, PLC" className="admin-input" />
                    </div>
                  </div>
                  <div className="flex gap-2 pt-1">
                    <button type="submit" className="btn-primary flex-1 py-2.5 text-xs flex items-center justify-center gap-1 cursor-pointer">
                      {editingProject ? 'Simpan' : 'Tambahkan'}
                    </button>
                    {editingProject && (
                      <button type="button" onClick={() => { setEditingProject(null); setProjectForm({ title: '', description: '', tags: '', year: '', category: 'Automation & IoT' }); }}
                        className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition">Batal</button>
                    )}
                  </div>
                </form>
              </div>
              <div className="lg:col-span-7 space-y-3">
                <h3 className="font-bold text-white">Daftar Proyek <span className="text-slate-500 text-sm font-normal">({projects.length})</span></h3>
                {projects.map(p => (
                  <div key={p.id} className="admin-card p-4 flex justify-between items-start gap-3 hover:border-slate-700 transition">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-[9px] font-mono bg-teal-500/10 text-teal-400 border border-teal-500/20 px-2 py-0.5 rounded">{p.year}</span>
                        <span className="text-[10px] text-slate-500">{p.category}</span>
                      </div>
                      <h4 className="font-semibold text-sm text-slate-100 truncate">{p.title}</h4>
                      <p className="text-[11px] text-slate-500 mt-0.5 line-clamp-1">{p.description}</p>
                    </div>
                    <div className="flex gap-1.5 shrink-0">
                      <button onClick={() => editProject(p)} className="p-2 rounded-lg bg-slate-800 hover:bg-teal-500/20 hover:text-teal-400 text-slate-400 transition"><EditIcon size={13}/></button>
                      <button onClick={() => deleteProject(p.id)} className="p-2 rounded-lg bg-slate-800 hover:bg-red-500/20 hover:text-red-400 text-slate-500 transition"><TrashIcon size={13}/></button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* EXPERIENCE TAB */}
          {activeDashboardTab === 'experience' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <div className="lg:col-span-5 admin-card p-6 h-fit">
                <h3 className="font-bold text-white mb-4">{editingExp ? '✏️ Edit Riwayat' : '+ Tambah Riwayat'}</h3>
                <form onSubmit={handleExpSubmit} className="space-y-3">
                  <div>
                    <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Tipe</label>
                    <select value={expForm.type} onChange={e => setExpForm({...expForm, type: e.target.value})} className="admin-input">
                      <option value="organization">Organisasi</option>
                      <option value="education">Pendidikan</option>
                    </select>
                  </div>
                  {[
                    { label: 'Jabatan / Program Studi', key: 'title', placeholder: 'D4 Teknologi Rekayasa Otomasi' },
                    { label: 'Organisasi / Institusi', key: 'institution', placeholder: 'HIMATRO UNDIP' },
                  ].map(f => (
                    <div key={f.key}>
                      <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">{f.label}</label>
                      <input type="text" value={expForm[f.key]} onChange={e => setExpForm({...expForm, [f.key]: e.target.value})} required placeholder={f.placeholder} className="admin-input" />
                    </div>
                  ))}
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Periode</label>
                      <input type="text" value={expForm.period} onChange={e => setExpForm({...expForm, period: e.target.value})} required placeholder="2025-Kini" className="admin-input" />
                    </div>
                    <div>
                      <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Order</label>
                      <input type="number" value={expForm.order} onChange={e => setExpForm({...expForm, order: e.target.value})} className="admin-input" />
                    </div>
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Detail (per baris)</label>
                    <textarea value={expForm.bullets} onChange={e => setExpForm({...expForm, bullets: e.target.value})} rows="3" className="admin-input resize-none" placeholder="Poin 1&#10;Poin 2" />
                  </div>
                  <div className="flex gap-2">
                    <button type="submit" className="btn-primary flex-1 py-2.5 text-xs cursor-pointer">{editingExp ? 'Simpan' : 'Tambah'}</button>
                    {editingExp && <button type="button" onClick={() => { setEditingExp(null); setExpForm({ type: 'organization', title: '', institution: '', period: '', bullets: '', order: 0 }); }} className="px-4 py-2 rounded-xl bg-slate-800 text-slate-300 text-xs">Batal</button>}
                  </div>
                </form>
              </div>
              <div className="lg:col-span-7 space-y-4">
                {['organization', 'education'].map(type => (
                  <div key={type}>
                    <h4 className="text-xs font-bold text-teal-400 uppercase tracking-widest font-mono mb-2">{type === 'organization' ? 'Organisasi' : 'Pendidikan'}</h4>
                    <div className="space-y-2">
                      {experiences.filter(e => e.type === type).map(eItem => (
                        <div key={eItem.id} className="admin-card p-3 flex justify-between items-center gap-3">
                          <div>
                            <p className="text-xs font-semibold text-slate-200">{eItem.title}</p>
                            <p className="text-[10px] text-slate-500">{eItem.institution} · {eItem.period}</p>
                          </div>
                          <div className="flex gap-1.5 shrink-0">
                            <button onClick={() => editExp(eItem)} className="p-1.5 rounded-lg bg-slate-800 hover:bg-teal-500/20 hover:text-teal-400 text-slate-400 transition"><EditIcon size={12}/></button>
                            <button onClick={() => deleteExp(eItem.id)} className="p-1.5 rounded-lg bg-slate-800 hover:bg-red-500/20 hover:text-red-400 text-slate-500 transition"><TrashIcon size={12}/></button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* PROFILE TAB */}
          {activeDashboardTab === 'profile' && (
            <div className="max-w-2xl admin-card p-6 rounded-2xl">
              <h3 className="font-bold text-white mb-6">✏️ Edit Profil Publik</h3>
              <form onSubmit={handleProfileUpdate} className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { label: 'Nama Lengkap', key: 'name' },
                    { label: 'Headline', key: 'title' },
                  ].map(f => (
                    <div key={f.key}>
                      <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1.5">{f.label}</label>
                      <input type="text" value={profile[f.key]} onChange={e => setProfile({...profile, [f.key]: e.target.value})} required className="admin-input" />
                    </div>
                  ))}
                </div>
                <div>
                  <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1.5">Summary</label>
                  <textarea value={profile.summary} onChange={e => setProfile({...profile, summary: e.target.value})} rows="4" className="admin-input resize-none" />
                </div>
                <div className="grid grid-cols-3 gap-4">
                  {[
                    { label: 'IPK', key: 'gpa', placeholder: '3.70 / 4.00' },
                    { label: 'Lokasi', key: 'location' },
                    { label: 'Email', key: 'email', type: 'email' },
                  ].map(f => (
                    <div key={f.key}>
                      <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1.5">{f.label}</label>
                      <input type={f.type || 'text'} value={profile[f.key]} onChange={e => setProfile({...profile, [f.key]: e.target.value})} placeholder={f.placeholder} className="admin-input" />
                    </div>
                  ))}
                </div>
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { label: 'LinkedIn URL', key: 'linkedin' },
                    { label: 'CV URL', key: 'cvUrl' },
                  ].map(f => (
                    <div key={f.key}>
                      <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1.5">{f.label}</label>
                      <input type="text" value={profile[f.key]} onChange={e => setProfile({...profile, [f.key]: e.target.value})} className="admin-input" />
                    </div>
                  ))}
                </div>
                <button type="submit" className="btn-primary py-2.5 px-6 text-sm cursor-pointer">Simpan Perubahan</button>
              </form>
            </div>
          )}

          {/* MESSAGES TAB */}
          {activeDashboardTab === 'messages' && (
            <div className="max-w-3xl space-y-3">
              <h3 className="font-bold text-white mb-4">Inbox <span className="text-slate-500 font-normal">({adminMessages.length} pesan)</span></h3>
              {adminMessages.length === 0 ? (
                <div className="admin-card p-16 text-center text-slate-600 text-sm rounded-2xl">
                  <MailIcon size={32} className="mx-auto mb-3 opacity-30" />
                  Belum ada pesan masuk.
                </div>
              ) : adminMessages.map(msg => (
                <div key={msg.id} className="admin-card p-5 flex justify-between items-start gap-4">
                  <div className="space-y-1.5 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-sm text-slate-100">{msg.name}</span>
                      <span className="text-[10px] text-slate-600 font-mono">{msg.createdAt?.seconds ? new Date(msg.createdAt.seconds * 1000).toLocaleString('id-ID') : 'Baru'}</span>
                    </div>
                    <div className="text-xs text-teal-400">{msg.email}</div>
                    <p className="text-xs text-slate-400 leading-relaxed bg-[#020817] p-3 rounded-lg border border-slate-800/60 mt-1">{msg.message}</p>
                  </div>
                  <button onClick={() => deleteMessage(msg.id)} className="p-2 rounded-lg bg-slate-800 hover:bg-red-500/20 hover:text-red-400 text-slate-500 transition shrink-0"><TrashIcon size={13}/></button>
                </div>
              ))}
            </div>
          )}
        </main>
      </div>
    );
  }

  // ======================== PUBLIC VIEW ========================
  return (
    <div className="min-h-screen bg-[#020817] text-slate-100 overflow-x-hidden">
      {/* Noise overlay */}
      <div className="bg-noise" />
      
      {/* Fixed background elements */}
      <div className="bg-grid fixed inset-0 pointer-events-none z-0" />
      <div className="fixed top-0 left-1/2 -translate-x-1/2 w-[900px] h-[500px] pointer-events-none z-0"
        style={{ background: 'radial-gradient(ellipse at center, rgba(20,184,166,0.1) 0%, rgba(16,185,129,0.05) 40%, transparent 70%)' }} />
      <div className="orb-blue fixed w-[600px] h-[600px] top-[20%] right-[-200px]" style={{zIndex:0}} />
      <div className="orb-emerald fixed w-[400px] h-[400px] bottom-[10%] left-[-150px]" style={{zIndex:0}} />

      {/* ======================== NAVBAR ======================== */}
      <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? 'glass-nav' : 'bg-transparent'}`}>
        <div className="max-w-6xl mx-auto px-6 py-4 flex justify-between items-center">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-teal-400 to-emerald-500 flex items-center justify-center font-extrabold text-slate-950 text-base shadow-lg">
                R
              </div>
              <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-teal-400 border-2 border-[#020817] led-blink" />
            </div>
            <div>
              <div className="font-bold text-base text-white leading-none">REVITA N.Y.P.</div>
              <div className="text-[9px] font-mono text-teal-400/70 tracking-widest">AUTOMATION ENGINEER</div>
            </div>
          </div>

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center gap-1">
            {navLinks.map(link => (
              <a key={link.href} href={link.href}
                className={`nav-link px-3 py-2 rounded-lg hover:bg-slate-800/50 transition-all ${activeSection === link.href.slice(1) ? 'text-teal-400' : ''}`}>
                {link.label}
              </a>
            ))}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2">
            {currentUser ? (
              <button onClick={() => setView('dashboard')} className="btn-primary px-4 py-2 text-xs flex items-center gap-1.5 cursor-pointer">
                <ZapIcon size={12} /> Dashboard
              </button>
            ) : (
              <button onClick={() => setView('login')} className="btn-ghost px-4 py-2 text-xs flex items-center gap-1.5 cursor-pointer">
                <ShieldIcon size={12} /> Admin
              </button>
            )}
            <a href={profile.cvUrl} target="_blank" rel="noreferrer"
              className="px-4 py-2 text-xs font-bold rounded-xl bg-teal-500 text-slate-950 hover:bg-teal-400 transition flex items-center gap-1.5">
              Buka CV <ExternalLinkIcon size={12} />
            </a>
            {/* Mobile Menu Button */}
            <button className="md:hidden p-2 rounded-lg text-slate-400 hover:text-teal-400 transition" onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>
              {mobileMenuOpen ? <XIcon size={20} /> : <MenuIcon size={20} />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden glass-nav border-t border-slate-800/60 px-6 py-4 space-y-1">
            {navLinks.map(link => (
              <a key={link.href} href={link.href} onClick={() => setMobileMenuOpen(false)}
                className="block px-3 py-2.5 text-sm text-slate-400 hover:text-teal-400 rounded-lg hover:bg-slate-800/50 transition">
                {link.label}
              </a>
            ))}
          </div>
        )}
      </nav>

      {/* ======================== MAIN CONTENT ======================== */}
      <main className="relative z-10">

        {/* ======================== HERO SECTION ======================== */}
        <section id="about" className="min-h-screen flex items-center pt-20">
          <div className="max-w-6xl mx-auto px-6 w-full py-20">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
              {/* Left */}
              <div className="space-y-8">
                {/* Status Badge */}
                <div className="status-badge w-fit">
                  <span className="w-2 h-2 rounded-full bg-teal-400 signal-pulse" />
                  {profile.title}
                </div>

                {/* Name */}
                <div>
                  <h1 className="text-5xl md:text-7xl font-black tracking-tighter leading-none">
                    <span className="text-white block">
                      {profile.name.split(' ').slice(0, 1).join(' ')}
                    </span>
                    <span className="text-white block">
                      {profile.name.split(' ').slice(1, 3).join(' ')}
                    </span>
                    <span className="heading-teal block text-4xl md:text-5xl mt-1">
                      {profile.name.split(' ').slice(3).join(' ')}
                    </span>
                  </h1>
                </div>

                {/* Summary */}
                <p className="text-slate-400 text-base leading-relaxed max-w-lg">
                  {profile.summary}
                </p>

                {/* Stats Row */}
                <div className="flex gap-6">
                  <div>
                    <div className="stat-number text-3xl">
                      <AnimatedCounter target={370} /><span className="text-lg text-teal-400">/400</span>
                    </div>
                    <div className="text-xs text-slate-500 mt-1 font-mono">IPK KUMULATIF</div>
                  </div>
                  <div className="w-px bg-slate-800" />
                  <div>
                    <div className="stat-number text-3xl">
                      <AnimatedCounter target={6} />
                    </div>
                    <div className="text-xs text-slate-500 mt-1 font-mono">SEMESTER AKTIF</div>
                  </div>
                  <div className="w-px bg-slate-800" />
                  <div>
                    <div className="stat-number text-3xl">
                      <AnimatedCounter target={projects.length || 8} suffix="+" />
                    </div>
                    <div className="text-xs text-slate-500 mt-1 font-mono">PROYEK SELESAI</div>
                  </div>
                </div>

                {/* Info chips */}
                <div className="flex flex-wrap gap-2">
                  <div className="flex items-center gap-2 tech-badge">
                    <MapPinIcon size={12} /> {profile.location}
                  </div>
                  <div className="flex items-center gap-2 tech-badge">
                    <GraduationCapIcon size={12} /> UNDIP 2023–2027
                  </div>
                  <div className="flex items-center gap-2 tech-badge">
                    <CpuIcon size={12} /> Automation & IoT
                  </div>
                </div>

                {/* CTA Buttons */}
                <div className="flex flex-wrap gap-3">
                  <a href="#contact" className="btn-primary px-6 py-3 text-sm flex items-center gap-2">
                    <SendIcon size={15} /> Hubungi Saya
                  </a>
                  <a href={profile.cvUrl} target="_blank" rel="noreferrer" className="btn-ghost px-6 py-3 text-sm flex items-center gap-2">
                    Lihat CV Lengkap <ExternalLinkIcon size={14} />
                  </a>
                </div>

                {/* Social links */}
                <div className="flex items-center gap-3">
                  {[
                    { href: profile.github, icon: <GithubIcon size={18} />, label: 'GitHub' },
                    { href: `https://${profile.linkedin}`, icon: <LinkedinIcon size={18} />, label: 'LinkedIn' },
                    { href: `mailto:${profile.email}`, icon: <MailIcon size={18} />, label: 'Email' },
                  ].map(s => (
                    <a key={s.label} href={s.href} target="_blank" rel="noreferrer"
                      className="flex items-center gap-2 p-3 rounded-xl bg-slate-900/60 border border-slate-800/60 text-slate-400 hover:text-teal-400 hover:border-teal-500/30 transition group">
                      {s.icon}
                    </a>
                  ))}
                  <div className="h-px flex-1 bg-gradient-to-r from-slate-800 to-transparent" />
                  <span className="text-[10px] text-slate-600 font-mono">{profile.email}</span>
                </div>
              </div>

              {/* Right - Terminal Card */}
              <div className="hidden lg:block">
                <div className="relative">
                  {/* Glow effect behind card */}
                  <div className="absolute inset-0 rounded-2xl blur-3xl" style={{ background: 'radial-gradient(ellipse, rgba(20,184,166,0.15) 0%, transparent 70%)' }} />
                  
                  <div className="relative rounded-2xl overflow-hidden border border-slate-800/60 shadow-2xl">
                    {/* Terminal header */}
                    <div className="terminal-header">
                      <span className="terminal-dot bg-red-500" />
                      <span className="terminal-dot bg-yellow-500" />
                      <span className="terminal-dot bg-green-500" />
                      <span className="flex-1 text-center text-[10px] text-slate-500 font-mono">revita@portfolio:~</span>
                    </div>
                    
                    {/* Terminal body */}
                    <div className="terminal-body relative min-h-[320px]">
                      <div className="scan-line" />
                      <div className="space-y-1">
                        <div><span className="text-teal-400">revita</span><span className="text-slate-600">@portfolio</span><span className="text-slate-400">:~$ </span><span className="text-slate-300">whoami</span></div>
                        <div className="text-emerald-400 ml-4">{profile.name}</div>
                        <div className="mt-3"><span className="text-teal-400">revita</span><span className="text-slate-600">@portfolio</span><span className="text-slate-400">:~$ </span><span className="text-slate-300">cat profile.json</span></div>
                        <div className="text-slate-400 ml-4 mt-1">
                          <div><span className="text-blue-400">{'{'}</span></div>
                          <div className="ml-4"><span className="text-yellow-400">"role"</span>: <span className="text-emerald-400">"Automation & IoT Engineer"</span>,</div>
                          <div className="ml-4"><span className="text-yellow-400">"university"</span>: <span className="text-emerald-400">"Universitas Diponegoro"</span>,</div>
                          <div className="ml-4"><span className="text-yellow-400">"gpa"</span>: <span className="text-orange-400">{profile.gpa}</span>,</div>
                          <div className="ml-4"><span className="text-yellow-400">"focus"</span>: [</div>
                          <div className="ml-8"><span className="text-emerald-400">"PLC & SCADA"</span>,</div>
                          <div className="ml-8"><span className="text-emerald-400">"Microcontroller"</span>,</div>
                          <div className="ml-8"><span className="text-emerald-400">"IoT Systems"</span></div>
                          <div className="ml-4">],</div>
                          <div className="ml-4"><span className="text-yellow-400">"status"</span>: <span className="text-emerald-400">"Open to opportunities"</span></div>
                          <div><span className="text-blue-400">{'}'}</span></div>
                        </div>
                        <div className="mt-3"><span className="text-teal-400">revita</span><span className="text-slate-600">@portfolio</span><span className="text-slate-400">:~$ </span><span className="cursor-blink text-slate-300">█</span></div>
                      </div>
                    </div>
                  </div>

                  {/* Floating stats around terminal */}
                  <div className="absolute -right-6 top-8 glass-card rounded-xl p-3 animate-float" style={{ animationDelay: '0s' }}>
                    <div className="text-[10px] text-slate-500 font-mono mb-0.5">SENSORS ACTIVE</div>
                    <div className="text-lg font-black heading-teal">12</div>
                  </div>
                  <div className="absolute -left-6 bottom-16 glass-card rounded-xl p-3 animate-float" style={{ animationDelay: '2s' }}>
                    <div className="text-[10px] text-slate-500 font-mono mb-0.5">UPTIME</div>
                    <div className="text-lg font-black heading-teal">99.7%</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ======================== SKILLS SECTION ======================== */}
        <section id="skills" className="py-24 section-divider">
          <div className="max-w-6xl mx-auto px-6">
            {/* Header */}
            <div className="text-center mb-16">
              <div className="section-label justify-center">keahlian teknis</div>
              <h2 className="text-4xl md:text-5xl font-black text-white mb-4">
                Tech Stack & Tools
              </h2>
              <p className="text-slate-500 max-w-xl mx-auto">
                Pemetaan kemampuan teknis, perangkat lunak yang dikuasai, dan kompetensi profesional.
              </p>
            </div>

            {/* Tab switcher */}
            <div className="flex justify-center mb-10">
              <div className="flex bg-slate-900/80 p-1.5 rounded-2xl border border-slate-800/60">
                {[
                  { id: 'technical', label: 'Automation Skills' },
                  { id: 'software', label: 'Tools & Software' },
                  { id: 'soft', label: 'Soft Skills' },
                ].map(tab => (
                  <button key={tab.id} onClick={() => setActiveSkillsTab(tab.id)}
                    className={`px-5 py-2.5 text-sm font-semibold rounded-xl transition-all ${activeSkillsTab === tab.id ? 'bg-teal-500 text-slate-950 shadow-lg' : 'text-slate-500 hover:text-slate-300'}`}>
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Technical Skills */}
            {activeSkillsTab === 'technical' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {[
                  { icon: <CpuIcon size={22} />, title: 'Automation Systems', desc: 'Merancang dan mengintegrasikan sistem kontrol otomasi menggunakan mikrokontroler Arduino & STM32 serta sistem kontrol PLC industri.', pct: 90 },
                  { icon: <TerminalIcon size={22} />, title: 'Programming Languages', desc: 'Menguasai C/C++ untuk mikroprosesor (Arduino IDE/STM32 CubeIDE) serta Delphi 7 untuk pengembangan antarmuka lokal.', pct: 85 },
                  { icon: <BriefcaseIcon size={22} />, title: 'Industrial Control & PLC', desc: 'Pengalaman perkabelan PLC, pemrograman ladder diagram, dan pemodelan proses melalui SimView.', pct: 80 },
                  { icon: <SlidersIcon size={22} />, title: 'Electronics & Circuit Design', desc: 'Merangkai sirkuit listrik, integrasi sensor-actuator analog & digital, serta analisis skematik via Proteus/Multisim.', pct: 88 },
                ].map((skill, i) => (
                  <div key={i} className="glass-card rounded-2xl p-6 relative overflow-hidden group">
                    <div className="absolute inset-0 bg-gradient-to-br from-teal-500/3 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                    <div className="relative">
                      <div className="flex items-center gap-3 mb-4">
                        <div className="p-2.5 rounded-xl bg-teal-500/10 text-teal-400 border border-teal-500/10">
                          {skill.icon}
                        </div>
                        <h3 className="font-bold text-base text-white">{skill.title}</h3>
                      </div>
                      <p className="text-slate-400 text-sm leading-relaxed mb-5">{skill.desc}</p>
                      <SkillBar label="Proficiency" pct={skill.pct} />
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Software Tools */}
            {activeSkillsTab === 'software' && (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {[
                  { name: 'Arduino IDE', category: 'Hardware', level: 'Expert' },
                  { name: 'LabVIEW', category: 'Control', level: 'Advanced' },
                  { name: 'Proteus', category: 'Simulation', level: 'Advanced' },
                  { name: 'SimView', category: 'PLC', level: 'Intermediate' },
                  { name: 'VS Code', category: 'Coding', level: 'Expert' },
                  { name: 'MatLab', category: 'Analysis', level: 'Intermediate' },
                  { name: 'STM32 CubeIDE', category: 'Embedded', level: 'Advanced' },
                  { name: 'AutoCAD', category: 'Design', level: 'Basic' },
                ].map((tool, i) => (
                  <div key={i} className="skill-chip group flex-col items-start gap-2">
                    <div className="flex items-center gap-2 w-full">
                      <span className="w-2 h-2 rounded-full bg-teal-400 group-hover:scale-125 transition-transform" />
                      <span className="font-semibold text-slate-200 text-sm">{tool.name}</span>
                    </div>
                    <div className="flex items-center justify-between w-full">
                      <span className="text-[10px] text-slate-500 font-mono">{tool.category}</span>
                      <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded font-mono ${
                        tool.level === 'Expert' ? 'bg-teal-500/15 text-teal-400' :
                        tool.level === 'Advanced' ? 'bg-emerald-500/15 text-emerald-400' :
                        tool.level === 'Intermediate' ? 'bg-blue-500/15 text-blue-400' :
                        'bg-slate-700 text-slate-400'
                      }`}>{tool.level}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Soft Skills */}
            {activeSkillsTab === 'soft' && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                  { title: 'Leadership', icon: '🎯', desc: 'Memimpin tim pengabdian masyarakat SDGs & pengurus aktif HIMATRO.' },
                  { title: 'Teamwork', icon: '🤝', desc: 'Berkolaborasi dengan tim multi-disiplin untuk mencapai tujuan bersama.' },
                  { title: 'Problem-Solving', icon: '🔬', desc: 'Menganalisis akar masalah teknik otomasi secara sistematis dan metodis.' },
                  { title: 'Communication', icon: '💬', desc: 'Menyampaikan konsep teknis kepada khalayak awam maupun juri akademis.' },
                ].map((skill, i) => (
                  <div key={i} className="glass-card rounded-2xl p-5 group hover:border-teal-500/25 transition-all">
                    <div className="text-3xl mb-3">{skill.icon}</div>
                    <div className="flex items-center gap-2 mb-2">
                      <CheckIcon size={14} className="text-emerald-400 shrink-0" />
                      <h3 className="font-bold text-sm text-teal-400">{skill.title}</h3>
                    </div>
                    <p className="text-slate-500 text-xs leading-relaxed">{skill.desc}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>

        {/* ======================== EXPERIENCE SECTION ======================== */}
        <section id="experience" className="py-24 section-divider">
          <div className="max-w-6xl mx-auto px-6">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 mb-16">
              <div>
                <div className="section-label">rekam jejak</div>
                <h2 className="text-4xl md:text-5xl font-black text-white">
                  Pendidikan &<br />Organisasi
                </h2>
                <p className="text-slate-500 text-sm mt-3 max-w-md">
                  Rekam jejak akademis dan kontribusi kepemimpinan dalam organisasi kemahasiswaan.
                </p>
              </div>
              <div className="flex bg-slate-900/80 p-1 rounded-xl border border-slate-800/60 shrink-0">
                {[
                  { id: 'organization', label: '👥 Organisasi' },
                  { id: 'education', label: '🎓 Pendidikan' },
                ].map(tab => (
                  <button key={tab.id} onClick={() => setActiveTimelineTab(tab.id)}
                    className={`py-2 px-4 text-xs font-bold rounded-lg transition-all ${activeTimelineTab === tab.id ? 'bg-teal-500 text-slate-950' : 'text-slate-500 hover:text-slate-300'}`}>
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="relative">
              <div className="timeline-line ml-4 md:ml-6" />
              <div className="ml-10 md:ml-14 space-y-6">
                {experiences.filter(e => e.type === activeTimelineTab).length === 0 ? (
                  <div className="glass-card p-10 rounded-2xl text-center text-slate-600 text-sm">
                    Belum ada data {activeTimelineTab === 'education' ? 'pendidikan' : 'organisasi'}.
                  </div>
                ) : (
                  experiences.filter(e => e.type === activeTimelineTab).map((eItem, i) => (
                    <div key={eItem.id} className="relative group">
                      <div className="timeline-dot" style={{ left: '-46px', top: '24px' }}>
                        <div className="absolute inset-0 rounded-full bg-teal-400 animate-pulse-glow opacity-0 group-hover:opacity-100 transition-opacity" />
                      </div>
                      <div className="glass-card rounded-2xl p-6 hover:border-teal-500/20 transition-all">
                        <div className="flex flex-col md:flex-row md:justify-between items-start md:items-center gap-3 mb-4">
                          <div>
                            <h3 className="font-extrabold text-lg text-white">{eItem.title}</h3>
                            <p className="text-teal-400 text-sm font-medium mt-0.5">{eItem.institution}</p>
                          </div>
                          <span className="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-[10px] font-bold text-slate-400 font-mono uppercase tracking-wider shrink-0">
                            {eItem.period}
                          </span>
                        </div>
                        {eItem.bullets?.length > 0 && (
                          <ul className="space-y-2">
                            {eItem.bullets.map((b, idx) => (
                              <li key={idx} className="flex items-start gap-2.5 text-sm text-slate-400">
                                <ChevronRightIcon size={14} className="text-teal-500 shrink-0 mt-0.5" />
                                <span>{b}</span>
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </section>

        {/* ======================== PROJECTS SECTION ======================== */}
        <section id="projects" className="py-24 section-divider">
          <div className="max-w-6xl mx-auto px-6">
            <div className="text-center mb-16">
              <div className="section-label justify-center">portofolio proyek</div>
              <h2 className="text-4xl md:text-5xl font-black text-white mb-4">Proyek Teknik</h2>
              <p className="text-slate-500 max-w-lg mx-auto text-sm">
                Koleksi proyek otomasi, instrumentasi, dan pemrograman yang dimuat langsung dari <span className="text-teal-400 font-mono">Cloud Firestore</span>.
              </p>
            </div>

            {/* Category filter */}
            <div className="flex flex-wrap justify-center gap-2 mb-10">
              {categories.map((cat, i) => (
                <button key={i} onClick={() => setActiveCategory(cat)}
                  className={`py-2 px-5 rounded-xl text-xs font-bold border transition-all ${
                    activeCategory === cat
                      ? 'bg-teal-500 text-slate-950 border-teal-500 shadow-lg shadow-teal-500/20'
                      : 'bg-slate-900/60 text-slate-500 border-slate-800/60 hover:border-slate-700 hover:text-slate-300'
                  }`}>
                  {cat}
                </button>
              ))}
            </div>

            {filteredProjects.length === 0 ? (
              <div className="glass-card p-20 rounded-2xl text-center">
                <div className="text-4xl mb-4">⚙️</div>
                <p className="text-slate-500 text-sm">Belum ada proyek untuk kategori ini.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
                {filteredProjects.map((project, i) => (
                  <div key={project.id} className="project-card p-6 flex flex-col">
                    {/* Accent top line */}
                    <div className="project-card-accent" />
                    
                    {/* Header */}
                    <div className="flex justify-between items-start mb-4">
                      <span className="text-[10px] font-mono font-bold text-teal-400 bg-teal-500/10 border border-teal-500/15 px-2.5 py-1 rounded-lg">
                        {project.year || 'PROYEK'}
                      </span>
                      <span className="text-[10px] text-slate-600 font-mono">{project.category}</span>
                    </div>

                    {/* Content */}
                    <h3 className="text-base font-bold text-slate-100 mb-3 leading-snug group-hover:text-teal-400 transition">
                      {project.title}
                    </h3>
                    <p className="text-slate-500 text-xs leading-relaxed flex-1 mb-5">{project.description}</p>

                    {/* Tags */}
                    <div className="flex flex-wrap gap-1.5 pt-4 border-t border-slate-800/40">
                      {project.tags?.map((tag, ti) => (
                        <span key={ti} className="tech-badge">{tag}</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>

        {/* ======================== CERTIFICATIONS SECTION ======================== */}
        <section id="certifications" className="py-24 section-divider">
          <div className="max-w-6xl mx-auto px-6">
            <div className="section-label">sertifikasi & prestasi</div>
            <div className="grid grid-cols-1 md:grid-cols-12 gap-8 mt-4">
              
              {/* Certifications */}
              <div className="md:col-span-7">
                <h2 className="text-3xl md:text-4xl font-black text-white mb-8 flex items-center gap-3">
                  <AwardIcon size={28} className="text-teal-400" />
                  Sertifikasi & Pelatihan
                </h2>
                
                <div className="cert-card p-6">
                  <div className="flex flex-col sm:flex-row sm:justify-between items-start gap-3 mb-5">
                    <div>
                      <div className="text-[10px] font-mono font-bold text-teal-400 tracking-widest mb-1.5">SERTIFIKAT RESMI</div>
                      <h3 className="font-extrabold text-base text-white leading-snug">Training on PLC, HMI, SCADA and IoT WECON</h3>
                      <p className="text-xs text-slate-500 mt-1 font-mono">Credential ID: WCA-GTC/II/2026</p>
                    </div>
                    <span className="px-3 py-1.5 bg-teal-500/10 text-teal-400 border border-teal-500/20 text-[10px] font-bold rounded-lg font-mono tracking-wider shrink-0">
                      LULUS 2026
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed mb-4">Pelatihan sertifikasi menyeluruh dalam ekosistem otomasi industrial WECON:</p>
                  <ul className="space-y-2.5">
                    {[
                      'Pemrograman logika PLC & perancangan HMI yang ramah pengguna.',
                      'Konfigurasi SCADA real-time monitoring dan integrasi modul IoT.',
                    ].map((item, i) => (
                      <li key={i} className="flex items-start gap-2.5 text-xs text-slate-400">
                        <CheckIcon size={14} className="text-teal-400 shrink-0 mt-0.5" />
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Achievements */}
              <div className="md:col-span-5">
                <h2 className="text-3xl md:text-4xl font-black text-white mb-8 flex items-center gap-3">
                  <GraduationCapIcon size={28} className="text-teal-400" />
                  Prestasi
                </h2>
                
                <div className="glass-card rounded-2xl p-6 relative overflow-hidden h-fit">
                  {/* Decorative */}
                  <div className="absolute top-0 right-0 w-32 h-32 rounded-full blur-3xl" style={{ background: 'radial-gradient(circle, rgba(20,184,166,0.1), transparent)' }} />
                  
                  <div className="relative">
                    <div className="inline-flex items-center gap-2 text-[10px] font-mono font-bold text-teal-400 tracking-widest bg-teal-500/10 border border-teal-500/15 px-3 py-1.5 rounded-lg mb-4">
                      <ZapIcon size={10} /> PROGRAM AKSELERASI
                    </div>
                    <h3 className="font-extrabold text-lg text-white mb-3 leading-snug">
                      Siswa Akselerasi SMAN 1 Magetan
                    </h3>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Terpilih sebagai satu dari <strong className="text-teal-400">13 siswa</strong> dalam kelompok akselerasi akademik untuk menuntaskan masa studi SMA dalam waktu lebih singkat.
                    </p>
                    <div className="mt-6 pt-4 border-t border-slate-800/60 flex items-center justify-between">
                      <span className="text-[10px] text-slate-600 font-mono">BATCH COHORT</span>
                      <span className="text-[10px] text-teal-400/70 font-mono font-bold">2021 — 2023</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ======================== CONTACT SECTION ======================== */}
        <section id="contact" className="py-24 section-divider">
          <div className="max-w-6xl mx-auto px-6">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
              
              {/* Left Info */}
              <div className="lg:col-span-5 space-y-8">
                <div>
                  <div className="section-label">hubungi saya</div>
                  <h2 className="text-4xl md:text-5xl font-black text-white leading-none">
                    Let's<br />
                    <span className="heading-teal">Connect.</span>
                  </h2>
                  <p className="text-slate-400 text-sm leading-relaxed mt-4 max-w-sm">
                    Tertarik berkolaborasi, berdiskusi proyek otomasi, atau merekrut saya? Jangan ragu untuk menghubungi!
                  </p>
                </div>

                {/* Contact cards */}
                <div className="space-y-3">
                  {[
                    { icon: <MailIcon size={18} />, label: 'Email', value: profile.email, href: `mailto:${profile.email}` },
                    { icon: <LinkedinIcon size={18} />, label: 'LinkedIn', value: profile.linkedin, href: `https://${profile.linkedin}` },
                    { icon: <MapPinIcon size={18} />, label: 'Lokasi', value: profile.location },
                  ].map((item, i) => (
                    <div key={i} className="glass-card rounded-xl p-4 flex items-center gap-4">
                      <div className="w-10 h-10 rounded-xl bg-teal-500/10 border border-teal-500/15 flex items-center justify-center text-teal-400 shrink-0">
                        {item.icon}
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">{item.label}</div>
                        {item.href ? (
                          <a href={item.href} target="_blank" rel="noreferrer" className="text-sm font-medium text-slate-200 hover:text-teal-400 transition">{item.value}</a>
                        ) : (
                          <span className="text-sm font-medium text-slate-200">{item.value}</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {/* QR placeholder */}
                <div className="glass-card rounded-xl p-4 flex items-center gap-4">
                  <div className="w-16 h-16 bg-white p-1.5 rounded-xl shrink-0">
                    <svg viewBox="0 0 100 100" className="w-full h-full text-slate-950">
                      <rect x="0" y="0" width="25" height="25" fill="black"/><rect x="5" y="5" width="15" height="15" fill="white"/>
                      <rect x="75" y="0" width="25" height="25" fill="black"/><rect x="80" y="5" width="15" height="15" fill="white"/>
                      <rect x="0" y="75" width="25" height="25" fill="black"/><rect x="5" y="80" width="15" height="15" fill="white"/>
                      <rect x="35" y="10" width="10" height="10" fill="black"/>
                      <rect x="55" y="25" width="10" height="10" fill="black"/>
                      <rect x="30" y="45" width="15" height="10" fill="black"/>
                      <rect x="60" y="50" width="10" height="15" fill="black"/>
                      <rect x="40" y="75" width="10" height="10" fill="black"/>
                      <rect x="80" y="65" width="10" height="10" fill="black"/>
                    </svg>
                  </div>
                  <div>
                    <div className="text-[10px] font-mono font-bold text-teal-400 tracking-widest mb-1">SCAN PORTFOLIO</div>
                    <p className="text-[11px] text-slate-500 leading-snug">Scan QR untuk melihat portofolio di smartphone Anda.</p>
                  </div>
                </div>
              </div>

              {/* Right Form */}
              <div className="lg:col-span-7">
                <div className="glass-card rounded-2xl p-8 relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-48 h-48 rounded-full blur-3xl pointer-events-none" style={{ background: 'radial-gradient(circle, rgba(20,184,166,0.08), transparent)' }} />
                  
                  <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                    <SendIcon size={18} className="text-teal-400" />
                    Kirim Pesan
                  </h3>

                  <form onSubmit={handleContactSubmit} className="space-y-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-[11px] font-bold text-slate-500 uppercase tracking-widest mb-2">Nama Lengkap</label>
                        <input type="text" name="name" value={formData.name} onChange={e => setFormData({...formData, name: e.target.value})} required placeholder="Nama Anda" className="form-input" />
                      </div>
                      <div>
                        <label className="block text-[11px] font-bold text-slate-500 uppercase tracking-widest mb-2">Email</label>
                        <input type="email" name="email" value={formData.email} onChange={e => setFormData({...formData, email: e.target.value})} required placeholder="email@anda.com" className="form-input" />
                      </div>
                    </div>
                    <div>
                      <label className="block text-[11px] font-bold text-slate-500 uppercase tracking-widest mb-2">Pesan</label>
                      <textarea name="message" value={formData.message} onChange={e => setFormData({...formData, message: e.target.value})} required rows="5" placeholder="Tulis pesan Anda di sini..." className="form-input resize-none" />
                    </div>
                    <button type="submit" className="btn-primary w-full py-3.5 text-sm flex items-center justify-center gap-2 cursor-pointer">
                      <SendIcon size={15} />
                      <span>Kirim Pesan</span>
                    </button>
                  </form>

                  {status && (
                    <div className={`mt-4 p-3.5 rounded-xl text-sm font-semibold text-center ${
                      statusType === 'success' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' :
                      statusType === 'error' ? 'bg-red-500/10 text-red-400 border border-red-500/20' :
                      'bg-teal-500/10 text-teal-400 border border-teal-500/20'
                    }`}>{status}</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </section>

      </main>

      {/* ======================== FOOTER ======================== */}
      <footer className="relative z-10 border-t border-slate-900/80 mt-12">
        <div className="max-w-6xl mx-auto px-6 py-10 flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal-400 to-emerald-500 flex items-center justify-center font-extrabold text-slate-950 text-sm">R</div>
            <div>
              <div className="text-sm font-bold text-white">Revita Nada Yuanika Putri</div>
              <div className="text-[10px] text-slate-600">© {new Date().getFullYear()} All Rights Reserved</div>
            </div>
          </div>
          <div className="text-center">
            <div className="text-[10px] text-slate-600 font-mono">Built with React · Tailwind CSS v4 · Firebase · Cloud Firestore</div>
          </div>
          <div className="flex items-center gap-3">
            {[
              { href: profile.github, icon: <GithubIcon size={16} /> },
              { href: `https://${profile.linkedin}`, icon: <LinkedinIcon size={16} /> },
              { href: `mailto:${profile.email}`, icon: <MailIcon size={16} /> },
            ].map((s, i) => (
              <a key={i} href={s.href} target="_blank" rel="noreferrer"
                className="p-2 rounded-lg text-slate-600 hover:text-teal-400 hover:bg-teal-500/10 transition">
                {s.icon}
              </a>
            ))}
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
