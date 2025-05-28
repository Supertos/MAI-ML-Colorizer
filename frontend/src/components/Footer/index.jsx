import styles from "./styles.module.scss";

export const Footer = () => {
  return (
    <div className={styles.footer}>
      <div className={styles.company}>© Supertos Industries</div>
      <div className={styles.email}>
        <a href="">supertos-industries@gmail.com</a>
      </div>
    </div>
  );
};
