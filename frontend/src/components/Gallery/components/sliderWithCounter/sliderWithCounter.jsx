import { useState } from "react";
import styles from "./styles.module.scss";

const SliderWithCounter = ({ label, description, id }) => {
  const [value, setValue] = useState(25);
  const [inputValue, setInputValue] = useState("25");

  const handleSliderChange = (e) => {
    const value = parseInt(e.target.value);
    setValue(value);
    setInputValue(value.toString());
  };

  const handleInputChange = (e) => {
    const value = e.target.value;
    setInputValue(value);

    if (/^\d+$/.test(value)) {
      const num = parseInt(value);
      if (num >= 0 && num <= 100) {
        setValue(num);
      }
    }
  };

  const handleBlur = () => {
    if (!/^\d+$/.test(inputValue)) {
      setInputValue(value.toString());
    }
  };

  const increment = () => {
    const newValue = Math.min(100, value + 1);
    setValue(newValue);
    setInputValue(newValue.toString());
  };

  const decrement = () => {
    const newValue = Math.max(0, value - 1);
    setValue(newValue);
    setInputValue(newValue.toString());
  };

  return (
    <label htmlFor={id} className={styles.sliderGroup}>
      <div className={styles.sliderRow}>
        <input
          id={id}
          type="range"
          min="0"
          max="100"
          value={value}
          onChange={handleSliderChange}
          className={styles.slider}
        />
        <div className={styles.counter}>
          <input
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            onBlur={handleBlur}
            className={styles.counterInput}
          />
          <div className={styles.buttons}>
            <button onClick={increment} className={styles.counterBtn}>
              ▲
            </button>
            <button onClick={decrement} className={styles.counterBtn}>
              ▼
            </button>
          </div>
        </div>
      </div>
      {label && <div className={styles.sliderLabel}>{label}</div>}
      {description && <div className={styles.description}>{description}</div>}
    </label>
  );
};

export default SliderWithCounter;
