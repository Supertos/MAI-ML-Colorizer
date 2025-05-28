import React from "react";
import styles from "./styles.module.scss";
import downloadIconUrl from "../../shared/assets/icons/downloadIcon.svg";
import closeIconUrl from "../../shared/assets/icons/closeIcon.svg";

export const PhotoCard = ({ photo, onClick, width, height, type }) => {
  return (
    <div className={styles.photo}>
      <div className={styles.card}>
        <img
          src={photo}
          alt="Uploaded"
          className={styles.image}
          width={width}
          height={height}
        />
        <button
          className={styles.deleteButton}
          onClick={onClick}
          aria-label="Удалить фотографию"
        >
          <img src={type === close ? closeIconUrl : downloadIconUrl} alt="" />
        </button>
      </div>
    </div>
  );
};
