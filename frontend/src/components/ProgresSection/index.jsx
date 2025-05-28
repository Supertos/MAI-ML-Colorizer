import React from "react";
import styles from "./styles.module.scss";

export const ProgressSection = ({
  progressPercentage,
  processedPhotos,
  totalPhotos,
  currentDateTime,
}) => {
  return (
    <div className={styles.downloadInformation}>
      <div className={styles.progressSection}>
        <div className={styles.progressBar}>
          <div
            className={styles.progressFill}
            style={{ width: `${progressPercentage}%` }}
          />
        </div>
        <div className={styles.photoCounter}>
          {processedPhotos}/{totalPhotos}
        </div>
      </div>
      <div className={styles.time}>{currentDateTime}</div>
    </div>
  );
};
