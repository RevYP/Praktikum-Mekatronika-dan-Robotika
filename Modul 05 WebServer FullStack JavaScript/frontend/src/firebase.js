import { initializeApp } from "firebase/app";
import { getAuth, connectAuthEmulator } from "firebase/auth";
import { getFirestore, connectFirestoreEmulator } from "firebase/firestore";

// Konfigurasi Client Firebase (dari portfolio-revita-036)
const firebaseConfig = {
  apiKey: "AIzaSyCB7lLc-0i2Ua4Ulnkp5nRTo1ABnGE6mQI",
  authDomain: "portfolio-revita-036.firebaseapp.com",
  projectId: "portfolio-revita-036",
  storageBucket: "portfolio-revita-036.firebasestorage.app",
  messagingSenderId: "425431648547",
  appId: "1:425431648547:web:eb3ea0cf05d5b78cb6e0cf"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);

// Hubungkan ke Emulator secara lokal di localhost
if (window.location.hostname === 'localhost') {
  connectAuthEmulator(auth, 'http://localhost:9099', { disableWarnings: true });
  connectFirestoreEmulator(db, 'localhost', 8085);
}

export { auth, db };
