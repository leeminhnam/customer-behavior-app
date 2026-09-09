import React, { useState } from 'react';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    model_type: 'logistic_regression',
    review_text:
      'This product exceeded my expectations! The texture is very lightweight, absorbs quickly, and improved my skin significantly within a week.',
    rating: 5,
    price_usd: 42.0,
    loves_count: 1250,
    helpful_votes: 18,
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'number' ? (value === '' ? '' : parseFloat(value)) : value,
    }));
  };

  const loadPreset = (type) => {
    if (type === 'positive') {
      setFormData((prev) => ({
        ...prev,
        review_text:
          'Absolutely love this serum! Gentle on sensitive skin, keeps me hydrated all day, and definitely worth every penny.',
        rating: 5,
        price_usd: 54.0,
        loves_count: 3200,
        helpful_votes: 42,
      }));
    } else if (type === 'negative') {
      setFormData((prev) => ({
        ...prev,
        review_text:
          'Terrible experience. The fragrance is overwhelmingly strong and caused redness immediately after application. Would not recommend.',
        rating: 1,
        price_usd: 68.0,
        loves_count: 150,
        helpful_votes: 5,
      }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const payload = {
      model_type: formData.model_type,
      review_text: formData.review_text,
      numerical_features: [
        formData.rating,
        formData.price_usd,
        formData.loves_count,
        formData.helpful_votes,
      ],
    };

    try {
      const response = await fetch('http://127.0.0.1:8000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Lỗi kết nối Server: HTTP ${response.status}`);
      }

      const data = await response.json();
      if (data.status === 'success') {
        setResult(data);
      } else {
        setError(data.message || 'Không thể thực hiện dự đoán, vui lòng kiểm tra lại!');
      }
    } catch (err) {
      setError(
        'Không thể kết nối tới FastAPI (http://127.0.0.1:8000). Hãy đảm bảo bạn đã chạy máy chủ backend FastAPI!'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ecom-app-layout">
      <header className="ecom-navbar">
        <div className="ecom-nav-container">
          <div className="ecom-logo-group">
            <span className="ecom-logo-icon" role="img" aria-label="cart">🛍️</span>
            <div className="ecom-logo-text">
              <h2>AI E-Commerce Insight</h2>
              <p>Phân tích Đa Phương Thức Ý Kiến & Dự Đoán Hành Vi Đề Xuất Khách Hàng</p>
            </div>
          </div>
          <div className="preset-container">
            <span className="preset-label">Mẫu phản hồi:</span>
            <div className="preset-buttons">
              <button
                type="button"
                className="btn-preset btn-preset-pos"
                onClick={() => loadPreset('positive')}
              >
                ✨ Đánh giá tích cực
              </button>
              <button
                type="button"
                className="btn-preset btn-preset-neg"
                onClick={() => loadPreset('negative')}
              >
                ⚠️ Đánh giá tiêu cực
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="ecom-main-grid">
        <section className="ecom-form-card">
          <form onSubmit={handleSubmit}>
            <div className="form-section-block">
              <h3 className="section-title">
                <span className="step-badge">0</span> Lựa Chọn Mô Hình (Model Selection)
              </h3>
              <div className="input-group">
                <label>Chọn Mô Hình Dự Đoán:</label>
                <div className="select-wrapper">
                  <select
                    name="model_type"
                    value={formData.model_type}
                    onChange={handleChange}
                    className="model-select"
                  >
                    <option value="logistic_regression">Logistic Regression (Machine Learning)</option>
                    <option value="dl">Deeper MLP (Deep Learning)</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="form-section-block">
              <h3 className="section-title">
                <span className="step-badge">1</span> Nội Dung Nhận Xét (Unstructured Review)
              </h3>
              <div className="input-group">
                <label>Ý kiến đánh giá / Review Text:</label>
                <textarea
                  name="review_text"
                  rows="4"
                  value={formData.review_text}
                  onChange={handleChange}
                  placeholder="Nhập cảm nhận chi tiết của khách hàng về sản phẩm..."
                  required
                />
                <span className="helper-text">
                  * Văn bản sẽ được chuẩn hóa NLP và trích xuất vector TF-IDF (3,000 chiều)
                </span>
              </div>
            </div>

            <div className="form-section-block">
              <h3 className="section-title">
                <span className="step-badge">2</span> Chỉ Số Định Lượng (Tabular Metrics)
              </h3>
              <div className="grid-responsive grid-2">
                <div className="input-group">
                  <label>Điểm đánh giá (Rating 1 - 5 sao):</label>
                  <input
                    type="number"
                    name="rating"
                    min="1"
                    max="5"
                    step="0.5"
                    value={formData.rating}
                    onChange={handleChange}
                    required
                  />
                </div>
                <div className="input-group">
                  <label>Giá sản phẩm (USD):</label>
                  <input
                    type="number"
                    name="price_usd"
                    min="0"
                    step="0.5"
                    value={formData.price_usd}
                    onChange={handleChange}
                    required
                  />
                </div>
                <div className="input-group">
                  <label>Lượt yêu thích (Loves Count):</label>
                  <input
                    type="number"
                    name="loves_count"
                    min="0"
                    step="1"
                    value={formData.loves_count}
                    onChange={handleChange}
                    required
                  />
                </div>
                <div className="input-group">
                  <label>Số lượt bình chọn hữu ích (Helpful Votes):</label>
                  <input
                    type="number"
                    name="helpful_votes"
                    min="0"
                    step="1"
                    value={formData.helpful_votes}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>
            </div>

            <button type="submit" className="btn-ecom-submit" disabled={loading}>
              {loading ? (
                <span className="btn-loading-wrapper">
                  <span className="mini-spinner"></span> Đang phân tích...
                </span>
              ) : (
                '🚀 Phân Tích Ý Kiến & Dự Đoán Hành Vi'
              )}
            </button>
          </form>
        </section>

        <section className="ecom-result-section">
          {error && (
            <div className="alert-card alert-error">
              <strong>⚠️ Lỗi hệ thống:</strong> {error}
            </div>
          )}

          {loading && (
            <div className="loading-card">
              <div className="spinner"></div>
              <p>Mô hình đang xử lý và suy luận...</p>
            </div>
          )}

          {!loading && !result && !error && (
            <div className="empty-card">
              <div className="empty-icon">📈</div>
              <h3>Sẵn sàng phân tích</h3>
              <p>
                Nhập nội dung nhận xét cùng các chỉ số tương tác bên trái hoặc chọn <b>Mẫu phản hồi</b> để trải nghiệm.
              </p>
            </div>
          )}

          {!loading && result && (
            <div
              className={`result-card ${
                result.prediction === 1 ? 'border-recommend' : 'border-not-recommend'
              }`}
            >
              <div className="result-header">
                <span
                  className={`badge-decision ${
                    result.prediction === 1 ? 'badge-pos' : 'badge-neg'
                  }`}
                >
                  {result.prediction === 1 ? '✅ SẼ ĐỀ XUẤT SẢN PHẨM' : '❌ KHÔNG ĐỀ XUẤT'}
                </span>
                <span className="timestamp">{new Date().toLocaleTimeString('vi-VN')}</span>
              </div>

              <div
                className={`decision-box ${
                  result.prediction === 1 ? 'box-pos' : 'box-neg'
                }`}
              >
                <div className="decision-sub-label">Dự đoán hành vi tiêu dùng</div>
                <div className="decision-main-label">{result.label}</div>
                <div className="decision-probability">
                  Xác suất đề xuất: <b>{result.recommendation_probability}%</b>
                </div>
              </div>

              <div className="confidence-wrapper">
                <div className="confidence-labels">
                  <span>Độ tin cậy mô hình ({result.model_used}):</span>
                  <b>{result.confidence}%</b>
                </div>
                <div className="confidence-track">
                  <div
                    className={`confidence-bar ${
                      result.prediction === 1 ? 'bar-pos' : 'bar-neg'
                    }`}
                    style={{ width: `${result.recommendation_probability}%` }}
                  ></div>
                </div>
              </div>

              <div className="summary-details">
                <div className="detail-item">
                  <span className="detail-label">Độ dài văn bản xử lý:</span>
                  <span className="detail-val">
                    {result.processed_text ? result.processed_text.split(' ').length : 0} từ
                  </span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Đánh giá người dùng:</span>
                  <span className="detail-val">⭐ {formData.rating} / 5.0</span>
                </div>
              </div>

              <div className="business-insight-box">
                <h4>💡 Đề xuất cho Nhà Bán Hàng:</h4>
                <p>
                  {result.prediction === 1
                    ? 'Khách hàng có trải nghiệm hài lòng. Nên khai thác đánh giá này vào danh sách nhận xét nổi bật để tối ưu tỷ lệ chuyển đổi đơn hàng.'
                    : 'Khách hàng bày tỏ sự thất vọng và rủi ro rời bỏ cao. Cần kích hoạt quy trình chăm sóc khách hàng tự động để hỗ trợ đổi trả hoặc tặng voucher bù đắp.'}
                </p>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;