// src/components/ResultLayout/index.jsx
import React, { useState, useEffect, useRef } from 'react';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';
import styles from './styles.module.scss';
import photoIconUrl from '../../shared/assets/icons/photoIcon.svg';
import { PhotoCard } from '../PhotoCard';
import { ProgressSection } from '../ProgresSection';
import { listImages } from '../../api';

export const ResultLayout = () => {
  const [images, setImages] = useState([]);
  const [processedPhotos, setProcessedPhotos] = useState(0);
  const pollingRef = useRef(null);

  useEffect(() => {
    let isMounted = true;

    const fetchImages = async () => {
      try {
        const imgs = await listImages();
        if (!isMounted) return;
        setImages(imgs);
        const doneCount = imgs.filter(img => img.inverted_download_url).length;
        setProcessedPhotos(doneCount);

        if (doneCount < imgs.length) {
          pollingRef.current = setTimeout(fetchImages, 3000);
        }
      } catch (e) {
        console.error('Ошибка при опросе результатов:', e);
      }
    };

    fetchImages();
    return () => {
      isMounted = false;
      if (pollingRef.current) clearTimeout(pollingRef.current);
    };
  }, []);

  const totalPhotos = images.length;
  const progressPercentage = totalPhotos
    ? Math.round((processedPhotos / totalPhotos) * 100)
    : 0;

  const currentDateTime = new Date().toLocaleTimeString('ru-RU', {
    hour: '2-digit',
    minute: '2-digit'
  });

  // Скачать один файл
  const handleDownload = (url, filename) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Упаковать все в ZIP и скачать
  const handleDownloadAll = async () => {
    if (!images.length) return;
    const zip = new JSZip();

    await Promise.all(
      images.map(async img => {
        const url = img.inverted_download_url || img.download_url;
        const response = await fetch(url);
        const blob = await response.blob();
        // Используем оригинальное имя файла с префиксом inverted_ для обработанных
        const filename = img.inverted_download_url
          ? `inverted_${img.filename}`
          : img.filename;
        zip.file(filename, blob);
      })
    );

    const content = await zip.generateAsync({ type: 'blob' });
    saveAs(content, 'restored_photos.zip');
  };

  return (
    <div className={styles.container}>
      <div className={styles.title}>
        <img src={photoIconUrl} alt="" />
        Ожидайте обработки
      </div>
      <div className={styles.titleDescription}>
        <p>Пожалуйста, подождите — ваш запрос обрабатывается</p>
      </div>

      <div className={styles.resultContainer}>
        <ProgressSection
          progressPercentage={progressPercentage}
          processedPhotos={processedPhotos}
          totalPhotos={totalPhotos}
          currentDateTime={currentDateTime}
        />

        <div className={styles.photosContainer}>
          {images.map(img => {
            const url = img.inverted_download_url || img.download_url;
            const filename = img.inverted_download_url
              ? `inverted_${img.filename}`
              : img.filename;
            return (
              <PhotoCard
                key={img.id}
                photo={url}
                width={258}
                height={329}
                type="download"
                onClick={() => handleDownload(url, filename)}
              />
            );
          })}
        </div>
      </div>

      <button
        className={styles.downloadAllButton}
        onClick={handleDownloadAll}
        type="button"
      >
        Скачать все
      </button>
    </div>
  );
};
