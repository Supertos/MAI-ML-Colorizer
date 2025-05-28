import styles from "./styles.module.scss";
import arrowDropDownUrl from "../../shared/assets/icons/arrowDropDown.svg";

export const Features = () => {
  return (
    <>
      <div className={styles.arrowIcon}>
        <img src={arrowDropDownUrl} alt="" />
      </div>

      <div className={styles.title}>Мы предлагаем</div>

      <div className={styles.features}>
        <div className={styles.group}>
          <h3 className={styles.subtitle}>Обычным пользователям</h3>
          <ul className={styles.list}>
            <li>
              <p className={styles.item}>Простой и удобный интерфейс</p>
              <p className={styles.itemDescription}>
                Простой ненагруженный интерфейс позволяет загружать и
                обрабатывать изобрежния даже бабушке!
              </p>
            </li>
            <li>
              <p className={styles.item}>Быстрая обработка изображений</p>
              <p className={styles.itemDescription}>
                Посредством использования множества машин Has-Been способен
                обрабатывать сотни запросов одновременно, гарантируя быстрое
                получение результата
              </p>
            </li>
          </ul>
        </div>

        <div className={styles.group}>
          <h3 className={styles.subtitle}>Историкам и архивистам</h3>
          <ul className={styles.list}>
            <li>
              <p className={styles.item}>Множественная обработка</p>
              <p className={styles.itemDescription}>
                Возможность загрузки и обработки множества изображений за раз
                позволяет тратить меньше времени на загрузку и больше - на
                работу с результатом
              </p>
            </li>
            <li>
              <p className={styles.item}>Настраиваемая ретушь</p>
              <p className={styles.itemDescription}>
                Возможность контролировать вмешательство сервиса в структуру
                изображения позволяет сохранить исходное качество изображения
              </p>
            </li>
            <li>
              <p className={styles.item}>Исторические наборы данных</p>
              <p className={styles.itemDescription}>
                Искуственный интеллект был обучен на исторических фото
                Российской Империи, что позволило повысить достоверность
                раскрашивания архивных фото
              </p>
            </li>
          </ul>
        </div>
      </div>
    </>
  );
};
