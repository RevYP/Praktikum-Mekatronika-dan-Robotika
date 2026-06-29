const admin = require('firebase-admin');

// Set host emulator Firestore ke port 8085 sesuai firebase.json kita
process.env.FIRESTORE_EMULATOR_HOST = '127.0.0.1:8085';

// Gunakan project ID yang sesuai dengan konfigurasi (.firebaserc / api.js)
const projectId = 'portfolio-revita-036';
admin.initializeApp({
  projectId: projectId
});

const db = admin.firestore();

const profile = {
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

const experiences = [
  {
    type: 'organization',
    title: "Team Leader - TRO's Student Group for SDGs",
    institution: 'Vocational School UNDIP',
    period: '2025',
    bullets: [
      'Memimpin program pemberdayaan masyarakat yang didanai di bawah inisiatif SDGs.',
      'Mendukung petani aren lokal di Desa Sriwulan, Kendal melalui penerapan efisiensi teknologi tepat guna.'
    ],
    order: 1
  },
  {
    type: 'organization',
    title: 'Head of Community Empowerment',
    institution: 'HIMATRO Sekolah Vokasi UNDIP',
    period: '2025 - SEKARANG',
    bullets: [
      'Mengawasi jalannya perencanaan dan eksekusi berbagai program pengabdian berfokus masyarakat.',
      'Berkoordinasi aktif dengan stakeholders eksternal untuk menjamin program yang berkelanjutan (sustainable).'
    ],
    order: 2
  },
  {
    type: 'organization',
    title: 'Chief Executive - Automation Inspiration (AI)',
    institution: 'HIMATRO Sekolah Vokasi UNDIP',
    period: '2024',
    bullets: [
      'Menjadi penanggung jawab utama program edukasi robotika dasar 2 hari untuk siswa Sekolah Dasar.',
      'Merevitalisasi fasilitas pusat kebudayaan setempat dan mengelola logistik, volunteer, serta anggaran dana program.'
    ],
    order: 3
  },
  {
    type: 'education',
    title: 'D4 Teknologi Rekayasa Otomasi',
    institution: 'Universitas Diponegoro (Semarang)',
    period: '2023 - 2027',
    bullets: [
      'Sedang menempuh jenjang sarjana terapan (Semester 6) dengan IPK kumulatif saat ini 3.70 / 4.00.',
      'Berfokus pada sistem kendali industri, mikroprosesor, IoT, instrumentasi boiler, serta fabrikasi mekanikal.'
    ],
    order: 4
  },
  {
    type: 'education',
    title: 'SMA Negeri 1 Magetan',
    institution: 'Jurusan Matematika & IPA (MIPA)',
    period: '2021 - 2023',
    bullets: [
      'Lulus dengan nilai rata-rata 93.82 / 100.',
      'Terpilih masuk sebagai siswa kelas program Akselerasi akademik (hanya 13 siswa terpilih per angkatan).'
    ],
    order: 5
  }
];

const projects = [
  {
    title: 'IoT-Based Automated Solar Panel Cleaning and Monitoring System',
    description: 'Developed an IoT-based solar panel cleaning and monitoring system using ESP32, INA219 sensors, and water pump automation for real-time monitoring and automatic cleaning. Programmed electrical systems and prepared the Bill of Materials (BOM) for project implementation.',
    tags: ['ESP32', 'IoT', 'Sensors', 'Automation', 'BOM Design'],
    year: '2026',
    category: 'Automation & IoT'
  },
  {
    title: 'Boiler Temperature Control System',
    description: 'Developed a boiler monitoring system using DS18B20 temperature sensor and JSN-SR04T ultrasonic sensor with Arduino Uno and LabVIEW visualization. Programmed sensor data acquisition and created a real-time monitoring interface in LabVIEW.',
    tags: ['Arduino Uno', 'LabVIEW', 'Sensors', 'Industrial Control'],
    year: '2025',
    category: 'Industrial Control'
  },
  {
    title: 'Mechanical Design & Fabrication',
    description: 'Executed bookshelf fabrication and assembly processes for the Mechanical Technology Practicum. Prepared the Bill of Materials (BOM) for project implementation.',
    tags: ['Mechanical Design', 'Fabrication', 'BOM Design'],
    year: '2024',
    category: 'Mechanical'
  },
  {
    title: 'Simple Library Management System',
    description: 'Developed a basic library management application using C language in Visual Studio Code IDE for Algorithms and Programming final project.',
    tags: ['C Language', 'VS Code', 'Data Management'],
    year: '2023',
    category: 'Programming'
  }
];

async function seedData() {
  console.log('Memulai seeding data ke Firestore Emulator...');
  try {
    // 1. Seed Projects
    const projCol = db.collection('projects');
    const projSnap = await projCol.get();
    let batch = db.batch();
    projSnap.docs.forEach((doc) => batch.delete(doc.ref));
    await batch.commit();
    console.log('Data projects lama dibersihkan.');

    for (const p of projects) {
      await projCol.add(p);
      console.log(`Berhasil menambahkan proyek: "${p.title}"`);
    }

    // 2. Seed Experience
    const expCol = db.collection('experience');
    const expSnap = await expCol.get();
    batch = db.batch();
    expSnap.docs.forEach((doc) => batch.delete(doc.ref));
    await batch.commit();
    console.log('Data experience lama dibersihkan.');

    for (const e of experiences) {
      await expCol.add(e);
      console.log(`Berhasil menambahkan riwayat: "${e.title}"`);
    }

    // 3. Seed Profile
    const profCol = db.collection('profile');
    await profCol.doc('default').set(profile);
    console.log('Data profil default berhasil diseed.');

    console.log('Seeding selesai dengan sukses!');
  } catch (error) {
    console.error('Gagal melakukan seeding data:', error);
  }
  process.exit();
}

seedData();
