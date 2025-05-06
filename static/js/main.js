// Ana JavaScript dosyası
document.addEventListener('DOMContentLoaded', function() {
    // Form Kontrolleri
    const multiStepForm = document.getElementById('multiStepForm');
    if (multiStepForm) {
        setupMultiStepForm();
    }

    // Grafikler için
    setupCharts();

    // Form validasyonu
    const forms = document.querySelectorAll('.needs-validation');
    if (forms.length > 0) {
        setupFormValidation(forms);
    }

    // Sonuç sayfası animasyonları
    const resultContainer = document.querySelector('.result-container');
    if (resultContainer) {
        resultContainer.classList.add('animate-fade-in');
    }

    // Tooltip aktivasyonu
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // İstatistik sayacı animasyonu
    animateStatCounters();
});

// Çok adımlı form ayarları
function setupMultiStepForm() {
    const formSections = document.querySelectorAll('.form-section');
    const nextButtons = document.querySelectorAll('.btn-next');
    const prevButtons = document.querySelectorAll('.btn-prev');
    const steps = document.querySelectorAll('.step');
    let currentSection = 0;

    // İlk bölümü göster
    formSections[0].classList.add('active');
    steps[0].classList.add('active');

    // İleri butonları
    nextButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Geçerli bölümün validasyonu
            const currentInputs = formSections[currentSection].querySelectorAll('input, select');
            let isValid = true;
            
            currentInputs.forEach(input => {
                if (!input.checkValidity()) {
                    isValid = false;
                    input.classList.add('is-invalid');
                } else {
                    input.classList.remove('is-invalid');
                }
            });
            
            if (!isValid) return;
            
            // Geçerli bölümü gizle ve sonrakini göster
            formSections[currentSection].classList.remove('active');
            currentSection++;
            formSections[currentSection].classList.add('active');
            
            // Adım göstergesini güncelle
            updateStepIndicator(currentSection);
        });
    });

    // Geri butonları
    prevButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Geçerli bölümü gizle ve öncekini göster
            formSections[currentSection].classList.remove('active');
            currentSection--;
            formSections[currentSection].classList.add('active');
            
            // Adım göstergesini güncelle
            updateStepIndicator(currentSection);
        });
    });

    // Adım göstergesini güncelle
    function updateStepIndicator(stepIndex) {
        steps.forEach((step, index) => {
            if (index < stepIndex) {
                step.classList.remove('active');
                step.classList.add('completed');
            } else if (index === stepIndex) {
                step.classList.remove('completed');
                step.classList.add('active');
            } else {
                step.classList.remove('active');
                step.classList.remove('completed');
            }
        });
    }
}

// Form validasyonu
function setupFormValidation(forms) {
    Array.prototype.slice.call(forms).forEach(function (form) {
        form.addEventListener('submit', function (event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
}

// İstatistik sayaçları animasyonu
function animateStatCounters() {
    const counters = document.querySelectorAll('.counter-value');
    
    counters.forEach(counter => {
        const target = parseInt(counter.getAttribute('data-target'));
        const duration = 2000; // ms cinsinden animasyon süresi
        const startTime = performance.now();
        
        function updateCounter(currentTime) {
            const elapsedTime = currentTime - startTime;
            const progress = Math.min(elapsedTime / duration, 1);
            const currentValue = Math.floor(progress * target);
            
            counter.textContent = currentValue.toLocaleString();
            
            if (progress < 1) {
                requestAnimationFrame(updateCounter);
            } else {
                counter.textContent = target.toLocaleString();
            }
        }
        
        requestAnimationFrame(updateCounter);
    });
}

// Grafikler
function setupCharts() {
    // Dashboard Grafikleri
    if (document.getElementById('churnRateChart')) {
        createChurnRateChart();
    }
    
    if (document.getElementById('featureImportanceChart')) {
        createFeatureImportanceChart();
    }
    
    if (document.getElementById('probabilityGauge')) {
        createProbabilityGauge();
    }
}

// Müşteri Kaybı Oranı Grafiği
function createChurnRateChart() {
    const ctx = document.getElementById('churnRateChart').getContext('2d');
    
    // Sunucudan veri almak için AJAX kullanabilirsiniz
    // Bu örnekte statik veriler kullanılıyor
    const data = {
        labels: ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran'],
        datasets: [{
            label: 'Müşteri Kaybı Oranı (%)',
            data: [12, 19, 15, 17, 14, 13],
            backgroundColor: 'rgba(66, 133, 244, 0.2)',
            borderColor: 'rgba(66, 133, 244, 1)',
            borderWidth: 2,
            tension: 0.4
        }]
    };
    
    new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Aylık Müşteri Kaybı Oranı'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 25
                }
            }
        }
    });
}

// Özellik Önem Grafiği
function createFeatureImportanceChart() {
    const ctx = document.getElementById('featureImportanceChart').getContext('2d');
    
    // Sunucudan veri almak için AJAX kullanabilirsiniz
    // Bu örnekte statik veriler kullanılıyor
    const data = {
        labels: ['Sözleşme Süresi', 'Aylık Ücret', 'Toplam Ücret', 'İnternet Hizmeti', 'Kullanım Süresi'],
        datasets: [{
            label: 'Özellik Önemi',
            data: [0.35, 0.28, 0.18, 0.12, 0.07],
            backgroundColor: [
                'rgba(66, 133, 244, 0.7)',
                'rgba(52, 168, 83, 0.7)',
                'rgba(251, 188, 5, 0.7)',
                'rgba(234, 67, 53, 0.7)',
                'rgba(138, 78, 159, 0.7)'
            ],
            borderWidth: 1
        }]
    };
    
    new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Özellik Önem Sıralaması'
                },
                legend: {
                    display: false
                }
            }
        }
    });
}

// Olasılık Göstergesi Grafiği
function createProbabilityGauge() {
    const ctx = document.getElementById('probabilityGauge').getContext('2d');
    
    // Müşteri kaybı olasılığı (sunucudan gelebilir)
    const probabilityValue = parseFloat(document.getElementById('probabilityGauge').getAttribute('data-value')) || 0.65;
    
    const gaugeChartText = {
        id: 'gaugeChartText',
        afterDatasetsDraw(chart, args, options) {
            const { ctx, data, chartArea: { top, bottom, left, right, width, height } } = chart;
            
            ctx.save();
            const xCenter = chart.getDatasetMeta(0).data[0].x;
            const yCenter = chart.getDatasetMeta(0).data[0].y;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.font = 'bold 16px Arial';
            ctx.fillStyle = probabilityValue > 0.5 ? '#ea4335' : '#34a853';
            ctx.fillText(`${Math.round(probabilityValue * 100)}%`, xCenter, yCenter);
            
            ctx.textBaseline = 'top';
            ctx.font = '14px Arial';
            ctx.fillStyle = '#666';
            ctx.fillText('Ayrılma Olasılığı', xCenter, yCenter + 5);
            ctx.restore();
        }
    };
    
    const data = {
        datasets: [{
            data: [probabilityValue, 1 - probabilityValue],
            backgroundColor: [
                probabilityValue > 0.5 ? 'rgba(234, 67, 53, 0.8)' : 'rgba(52, 168, 83, 0.8)',
                'rgba(220, 220, 220, 0.5)'
            ],
            borderWidth: 0,
            cutout: '75%',
            circumference: 180,
            rotation: 270
        }]
    };
    
    new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: false
                }
            }
        },
        plugins: [gaugeChartText]
    });
}