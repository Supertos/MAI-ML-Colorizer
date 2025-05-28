// src/components/Gallery/index.jsx
import React, { useState } from "react";
import styles from "./styles.module.scss";
import SliderWithCounter from "/src/components/Gallery/components/sliderWithCounter/sliderWithCounter.jsx";
import photoIconUrl from "../../shared/assets/icons/photoIcon.svg";
import { PhotoCard } from "../PhotoCard";
import { UploadPhoto } from "./components/UploadPhoto";
import { useNavigate } from "react-router-dom";
import { uploadImage } from "../../api";

export const Gallery = () => {
  const navigate = useNavigate();

  // стейт для сырых File-объектов
  const [files, setFiles] = useState([]);
  // стейт для preview-URL'ов
  const [uploadedPhotos, setUploadedPhotos] = useState([]);
  const [grain, setGrain] = useState(25);
  const [sharpness, setSharpness] = useState(25);
  const [processed, setProcessed] = useState(0);

  // при выборе файлов — сохраняем и File[], и preview
  const handleUpload = (e) => {
    const chosenFiles = Array.from(e.target.files);
    setFiles((prev) => [...prev, ...chosenFiles]);
    const previews = chosenFiles.map((f) => URL.createObjectURL(f));
    setUploadedPhotos((prev) => [...prev, ...previews]);
    setProcessed(0);
  };

  // клик на Обработать — отправляем все файлы, по одному
  const handleProcess = async () => {
    if (files.length === 0) {
      alert("Сначала выберите фото");
      return;
    }

    let currentId = null;

    // 1) первый файл — чтобы получить anonymous_id
    try {
      const res = await uploadImage(files[0]);
      currentId = res.anonymous_id;
      setProcessed(1);
    } catch (e) {
      alert("Ошибка при загрузке первого файла: " + e.message);
      return;
    }

    // 2) остальные файлы в ряд
    for (let i = 1; i < files.length; i++) {
      try {
        await uploadImage(files[i]);
      } catch (_) {
        console.error("Не удалось загрузить", files[i].name);
      }
      setProcessed((p) => p + 1);
    }

    // 3) переход на результат
    navigate("/result");
  };

  // удаление превью и File из списка
  const handleRemove = (index) => {
    setUploadedPhotos((prev) => prev.filter((_, i) => i !== index));
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  return (
    <div className={styles.galleryContainer}>
      <div className={styles.title}>
        <img src={photoIconUrl} alt="" />
        Старые фото — новая жизнь
      </div>
      <div className={styles.titleDescription}>
        <p>Вернуть к жизни старые фотографии? Легко!</p>
        <p>Придайте старым фото яркие цвета с помощью</p>
        <p>нашего сервиса!</p>
      </div>

      <div className={styles.uploadSection}>
        <UploadPhoto onUpload={handleUpload} />

        <div className={styles.controls}>
          <div className={styles.titleControls}>Настройте обработку</div>
          <SliderWithCounter
            id="noiseReduction"
            label="Удаление шумов"
            value={grain}
            onChange={setGrain}
          />
          <SliderWithCounter
            id="contrast"
            label="Контраст"
            value={sharpness}
            onChange={setSharpness}
          />
          <br></br>
          <button
            className={styles.processButton}
            onClick={handleProcess}
            type="button"
          >
            Обработать {files.length > 0 && `(${files.length})`}
          </button>
        </div>
      </div>

      <div className={styles.titlePhotosContainer}>Загруженные фото</div>

      <div className={styles.photosContainer}>
        {uploadedPhotos.map((photo, index) => (
          <PhotoCard
            key={index}
            photo={photo}
            width={120}
            height={150}
            type={close}
            onClick={() => handleRemove(index)}
          />
        ))}
      </div>

      {files.length > 0 && (
        <div style={{ marginTop: "1rem", textAlign: "center", color: "#fff" }}>
          Загружено: {processed}/{files.length}
        </div>
      )}
    </div>
  );
};
