import axios from 'axios';

// Konfigurasi URL base API untuk local emulator dan production cloud.
// project ID: portfolio-revita-036
const API_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:5001/portfolio-revita-036/us-central1/api/api'
  : '/api';

const api = axios.create({
  baseURL: API_URL,
});

export default api;
