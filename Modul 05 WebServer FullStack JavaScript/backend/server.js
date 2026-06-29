const express = require('express');
const cors = require('cors');
const admin = require('firebase-admin');
const { FieldValue } = require('firebase-admin/firestore');
const functions = require('firebase-functions');
require('dotenv').config();

// Inisialisasi Firebase Admin
if (admin.apps.length === 0) {
    admin.initializeApp();
}
const db = admin.firestore();

const app = express();

// Middleware
app.use(cors({ origin: true }));
app.use(express.json());

// Middleware untuk mengecek otentikasi Firebase ID Token
const checkAuth = async (req, res, next) => {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({ success: false, error: 'Unauthorized: No token provided' });
    }
    const token = authHeader.split('Bearer ')[1];
    try {
        const decodedToken = await admin.auth().verifyIdToken(token);
        req.user = decodedToken;
        next();
    } catch (error) {
        return res.status(401).json({ success: false, error: 'Unauthorized: Invalid token' });
    }
};

// --- ENDPOINT API ---

// 1. GET: Ambil status server
app.get('/api/hello', (req, res) => {
    res.status(200).json({
        success: true,
        message: 'Backend Express Server berhasil terhubung!'
    });
});

// 2. PROFILE CRUD
// GET: Mengambil profile dinamis (Public)
app.get('/api/profile', async (req, res) => {
    try {
        const doc = await db.collection('profile').doc('default').get();
        if (!doc.exists) {
            const defaultProfile = {
                name: 'REVITA NADA YUANIKA PUTRI',
                title: 'AUTOMATION & IOT ENGINEERING STUDENT',
                summary: 'Highly motivated Sixth-semester student of Automation Engineering Technology at Diponegoro University with a 3.70/4.00 GPA. Skilled in developing automation systems, integrating hardware and software, and applying microcontroller-based technology. Experienced in leading teams for community empowerment and technical projects, with a strong commitment to delivering innovative solutions.',
                gpa: '3.70 / 4.00',
                location: 'Semarang, ID',
                email: 'revitanada05@gmail.com',
                linkedin: 'www.linkedin.com/in/revitanada',
                cvUrl: 'https://bit.ly/4npMvsR',
                github: 'https://github.com'
            };
            return res.status(200).json({ success: true, data: defaultProfile });
        }
        res.status(200).json({ success: true, data: doc.data() });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// PUT: Memperbarui profile (Protected)
app.put('/api/profile', checkAuth, async (req, res) => {
    try {
        const profileData = req.body;
        await db.collection('profile').doc('default').set(profileData, { merge: true });
        res.status(200).json({ success: true, message: 'Profil berhasil diperbarui!' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// 3. EXPERIENCE CRUD
// GET: Mengambil riwayat (Public)
app.get('/api/experience', async (req, res) => {
    try {
        const snapshot = await db.collection('experience').orderBy('order', 'asc').get();
        const experiences = [];
        snapshot.forEach((doc) => {
            experiences.push({ id: doc.id, ...doc.data() });
        });
        res.status(200).json({ success: true, data: experiences });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// POST: Tambah riwayat (Protected)
app.post('/api/experience', checkAuth, async (req, res) => {
    try {
        const { type, title, institution, period, bullets, order } = req.body;
        if (!type || !title || !institution || !period) {
            return res.status(400).json({ success: false, error: 'Tipe, judul, institusi, dan periode wajib diisi!' });
        }
        const newExp = {
            type,
            title,
            institution,
            period,
            bullets: bullets || [],
            order: order !== undefined ? Number(order) : 0,
            createdAt: FieldValue.serverTimestamp()
        };
        const docRef = await db.collection('experience').add(newExp);
        res.status(201).json({ success: true, message: 'Riwayat berhasil ditambahkan!', id: docRef.id });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// PUT: Edit riwayat (Protected)
app.put('/api/experience/:id', checkAuth, async (req, res) => {
    try {
        const { id } = req.params;
        const updateData = req.body;
        if (updateData.order !== undefined) {
            updateData.order = Number(updateData.order);
        }
        await db.collection('experience').doc(id).update(updateData);
        res.status(200).json({ success: true, message: 'Riwayat berhasil diperbarui!' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// DELETE: Hapus riwayat (Protected)
app.delete('/api/experience/:id', checkAuth, async (req, res) => {
    try {
        const { id } = req.params;
        await db.collection('experience').doc(id).delete();
        res.status(200).json({ success: true, message: 'Riwayat berhasil dihapus!' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// 4. PROJECTS CRUD
// GET: Mengambil semua proyek (Public)
app.get('/api/projects', async (req, res) => {
    try {
        const snapshot = await db.collection('projects').get();
        const projects = [];
        snapshot.forEach((doc) => {
            projects.push({ id: doc.id, ...doc.data() });
        });
        res.status(200).json({ success: true, data: projects });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// POST: Tambah proyek (Protected)
app.post('/api/projects', checkAuth, async (req, res) => {
    try {
        const { title, description, tags, year, category } = req.body;
        if (!title || !description) {
            return res.status(400).json({ success: false, error: 'Judul dan deskripsi proyek wajib diisi!' });
        }
        const newProject = {
            title,
            description,
            tags: tags || [],
            year: year || '',
            category: category || 'General',
            createdAt: FieldValue.serverTimestamp()
        };
        const docRef = await db.collection('projects').add(newProject);
        res.status(201).json({ success: true, message: 'Proyek berhasil ditambahkan!', id: docRef.id });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// PUT: Edit proyek (Protected)
app.put('/api/projects/:id', checkAuth, async (req, res) => {
    try {
        const { id } = req.params;
        const updateData = req.body;
        await db.collection('projects').doc(id).update(updateData);
        res.status(200).json({ success: true, message: 'Proyek berhasil diperbarui!' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// DELETE: Hapus proyek (Protected)
app.delete('/api/projects/:id', checkAuth, async (req, res) => {
    try {
        const { id } = req.params;
        await db.collection('projects').doc(id).delete();
        res.status(200).json({ success: true, message: 'Proyek berhasil dihapus!' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// 5. CONTACT FORM & MESSAGES
// POST: Kirim pesan kontak (Public)
app.post('/api/contact', async (req, res) => {
    try {
        const { name, email, message } = req.body;

        if (!name || !email || !message) {
            return res.status(400).json({
                success: false,
                error: 'Nama, email, dan pesan wajib diisi!'
            });
        }

        const newMessage = {
            name,
            email,
            message,
            createdAt: FieldValue.serverTimestamp()
        };

        const docRef = await db.collection('messages').add(newMessage);
        res.status(201).json({
            success: true,
            message: 'Pesan berhasil dikirim!',
            id: docRef.id
        });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// GET: Ambil pesan masuk (Protected)
app.get('/api/messages', checkAuth, async (req, res) => {
    try {
        const snapshot = await db.collection('messages').orderBy('createdAt', 'desc').get();
        const messages = [];
        snapshot.forEach((doc) => {
            messages.push({ id: doc.id, ...doc.data() });
        });
        res.status(200).json({ success: true, data: messages });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// DELETE: Hapus pesan (Protected)
app.delete('/api/messages/:id', checkAuth, async (req, res) => {
    try {
        const { id } = req.params;
        await db.collection('messages').doc(id).delete();
        res.status(200).json({ success: true, message: 'Pesan berhasil dihapus!' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// --- RUN SERVER SECARA LOKAL (Jika tidak di-deploy ke Cloud Functions) ---
if (process.env.NODE_ENV !== 'production' && !process.env.FIREBASE_CONFIG) {
    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
        console.log(`Server lokal Express berjalan di http://localhost:${PORT}`);
    });
}

// --- EXPORT UNTUK FIREBASE FUNCTIONS ---
exports.api = functions.https.onRequest(app);
