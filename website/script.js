// 当页面滚动时，高亮当前部分在导航栏中的对应链接
document.addEventListener("DOMContentLoaded", function () {
  // 获取所有部分和导航链接
  const sections = document.querySelectorAll(".section");
  const navLinks = document.querySelectorAll("nav a");

  // 滚动监听
  window.addEventListener("scroll", function () {
    let current = "";

    sections.forEach((section) => {
      // 获取部分顶部距离视口顶部的距离
      const sectionTop = section.offsetTop;
      const sectionHeight = section.clientHeight;

      // 如果我们已经滚动到部分的顶部附近，将其设为当前部分
      if (pageYOffset >= sectionTop - 300) {
        current = section.getAttribute("id");
      }
    });

    // 移除所有导航链接的活动类
    navLinks.forEach((link) => {
      link.classList.remove("active");

      // 如果链接与当前部分匹配，添加活动类
      if (link.getAttribute("href").substring(1) === current) {
        link.classList.add("active");
      }
    });
  });

  // 添加平滑滚动效果
  navLinks.forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();

      const targetId = this.getAttribute("href");
      const targetSection = document.querySelector(targetId);

      window.scrollTo({
        top: targetSection.offsetTop,
        behavior: "smooth",
      });
    });
  });

  // 在页面加载后添加淡入效果
  document.body.classList.add("loaded");
});

// 添加图像点击放大查看功能
document.addEventListener("DOMContentLoaded", function () {
  const fullWidthImages = document.querySelectorAll(".full-width-img");

  fullWidthImages.forEach((img) => {
    img.addEventListener("click", function () {
      const modal = document.createElement("div");
      modal.classList.add("image-modal");

      const modalImg = document.createElement("img");
      modalImg.src = this.src;

      modal.appendChild(modalImg);
      document.body.appendChild(modal);

      // 点击模态框关闭它
      modal.addEventListener("click", function () {
        modal.classList.add("fade-out");
        setTimeout(() => {
          document.body.removeChild(modal);
        }, 300);
      });
    });
  });

  // 添加模态框样式
  const style = document.createElement("style");
  style.innerHTML = `
        .image-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1001;
            opacity: 0;
            animation: fadeIn 0.3s forwards;
            cursor: pointer;
        }

        .image-modal img {
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
            border: 2px solid white;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
        }

        .fade-out {
            animation: fadeOut 0.3s forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes fadeOut {
            from { opacity: 1; }
            to { opacity: 0; }
        }

        .loaded {
            animation: pageFadeIn 1s ease;
        }

        @keyframes pageFadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        nav a.active {
            color: #99ccff;
            border-bottom: 2px solid #99ccff;
        }
    `;
  document.head.appendChild(style);
});

// 处理移动端菜单
document.addEventListener("DOMContentLoaded", function () {
  const menuToggle = document.getElementById("menuToggle");
  const navMenu = document.getElementById("navMenu");
  const menuOverlay = document.getElementById("menuOverlay");

  if (menuToggle && navMenu && menuOverlay) {
    // 点击汉堡按钮打开菜单
    menuToggle.addEventListener("click", function () {
      navMenu.classList.toggle("show");
      menuOverlay.classList.toggle("show");
      document.body.style.overflow = navMenu.classList.contains("show")
        ? "hidden"
        : "";
    });

    // 点击菜单项关闭菜单
    navMenu.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", function () {
        navMenu.classList.remove("show");
        menuOverlay.classList.remove("show");
        document.body.style.overflow = "";
      });
    });

    // 点击遮罩层关闭菜单
    menuOverlay.addEventListener("click", function () {
      navMenu.classList.remove("show");
      menuOverlay.classList.remove("show");
      document.body.style.overflow = "";
    });
  }
});
