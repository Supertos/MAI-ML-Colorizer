// src/components/Gallery/components/UploadPhoto/index.jsx
import React from 'react';
import styles from './styles.module.scss';
import uploadIconUrl from '../../../../shared/assets/icons/uploadIcon.svg';

export const UploadPhoto = ({ onUpload }) => (
  <label className={styles.uploadBox}>
    <input
      type="file"
      accept="image/png,image/jpeg"
      multiple
      hidden
      onChange={(e) => onUpload(e)}
    />
    <img src={uploadIconUrl} alt="Upload" width={48} height={48} />
    <span>Загрузите фотографию</span>
  </label>
);
