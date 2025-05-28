// frontend/src/api.js
import { getCookie, setCookie } from './cookie';

export async function uploadImage(file) {
  let anonymousId = getCookie('anonymous_id');

  const form = new FormData();
  form.append('file', file);
  form.append('grain', 0);
  form.append('sharpness', 0);
  if (anonymousId) {
    form.append('anonymous_id', anonymousId);
  }

  const res = await fetch('http://localhost:8080/upload', {
    method: 'POST',
    body: form
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail?.message || 'Ошибка при загрузке');
  }

  const data = await res.json();
  if (!anonymousId && data.anonymous_id) {
    setCookie('anonymous_id', data.anonymous_id);
  }
  return data; // { image_id, original_filename, s3_key, anonymous_id }
}

export async function listImages() {
  const anonymousId = getCookie('anonymous_id');
  const res = await fetch(`http://localhost:8080/images/${anonymousId}`);
  if (!res.ok) throw new Error('Не удалось получить список изображений');
  return res.json(); // [{…}, …]
}
