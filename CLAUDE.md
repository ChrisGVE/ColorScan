# CLAUDE.md - Scan Colors Project

## SYSTEM RULES

See CLAUDE_SYSTEM.md for core directives and workflow.

## PROJECT CONTEXT

### Current Focus
Setting up Rust fountain pen ink color analysis crate with D65 anchor point calibration and comprehensive color measurement pipeline.

### Outstanding Decisions Needed
- [ ] OpenCV version compatibility verification for Rust bindings
- [ ] Choice between delta_e vs empfindung crate for color differences
- [ ] CLI argument structure and output format details

### Unconfirmed Assumptions
- [ ] 100ms performance target is achievable with OpenCV operations on smartphone images
- [ ] D65 anchor point provides sufficient accuracy across smartphone camera variations
- [ ] 10-15% minimum swatch size is practical for real-world fountain pen photos

### Project Structure

```
scan_colors/
├── CLAUDE_SYSTEM.md     # Shared system instructions (immutable)
├── CLAUDE.md            # This file - project tracking
├── PRD.txt              # Project requirements (enhanced with research)
├── ALGO.md              # Algorithm repository (color calibration algorithms)
├── src/
│   ├── lib.rs              # Main library interface
│   ├── constants.rs        # D65 illuminant & calibration parameters
│   ├── calibration/        # White balance & color correction
│   ├── detection/          # Paper & swatch detection
│   ├── color/             # Color analysis & conversion
│   ├── exif/             # EXIF metadata handling
│   └── error.rs          # Error types and handling
├── assets/
│   ├── test_samples/     # Unit test images with known properties
│   └── references/       # Reference swatches for validation
├── examples/             # CLI tool and usage examples
├── benches/             # Performance benchmarks
├── tests/               # Integration tests
└── docs/                # Additional documentation
```

### MCP Tool Status

- [x] Serena initial instructions attempted (needs source files)
- [ ] Claude-context codebase indexed (pending source creation)
- [x] Task-master system initialized
- [x] Research findings stored in memory knowledge graph
- [ ] Task-master tasks generated from PRD (AI service failed, using manual todo)

## TRACKING

### Todo

```markdown
- [x] Complete initial project research and PRD enhancement
  - [x] Research D65 illuminant standards and color calibration (2024)
  - [x] Research smartphone camera color matrix and calibration systems
  - [x] Research Adobe DNG profile system and license-free resources
  - [x] Store research findings in memory knowledge graph
  - [x] Update PRD.txt with comprehensive technical specifications
- [x] Initialize project structure
  - [x] Initialize git repository
  - [x] Copy template files (CLAUDE_SYSTEM.md, CLAUDE.md, ALGO.md)
  - [x] Initialize Rust crate with cargo
  - [x] Initialize task-master system
- [x] Create basic project directory structure
  - [x] Create src/ module structure (calibration/, detection/, color/, exif/)
  - [x] Create assets/ directory with test_samples/ and references/ subdirs
  - [x] Create examples/, benches/, tests/, docs/ directories
- [x] Setup Cargo.toml with enhanced dependency stack
  - [x] Add image processing dependencies (image, imageproc, opencv)
  - [x] Add color science dependencies (palette, empfindung, lab, rgb)
  - [x] Add EXIF handling (kamadak-exif)
  - [x] Configure opencv features for buildtime-bindgen
- [x] Implement core library structure
  - [x] Define ColorResult and AnalysisError types in lib.rs
  - [x] Create constants.rs with D65 illuminant and calibration parameters
  - [x] Setup error.rs with comprehensive error handling
  - [x] Create module structure with proper exports
- [x] Create basic CLI tool in examples/
  - [x] Simple file input → stdout JSON output
  - [x] Basic argument parsing and error handling
- [ ] Setup comprehensive testing framework
  - [ ] Integration tests structure
  - [ ] Performance benchmark framework
  - [ ] Sample image collection strategy
- [x] Initial commit and documentation
  - [x] Complete project setup commit
  - [x] Create basic README.md with usage examples
  - [x] Document research findings in ALGO.md
```

### Memory Tags
- D65_Illuminant_Research: D65 standard illuminant specifications and anchor point implementation
- Smartphone_Color_Matrix_2024: Latest smartphone color calibration developments
- Adobe_DNG_Profile_System: License-free camera profile resources
- ColorChecker_Calibration: Industry standard calibration workflows
- Color_Constancy_Chromatic_Adaptation: Algorithms for illuminant adaptation

### Progress Log (include date and local time)

**2025-01-19 16:45 EST**: ✅ **Project Research & PRD Enhancement Complete**

Comprehensive research completed on color calibration standards and smartphone camera systems:

🔬 **Research Findings:**
- D65 illuminant confirmed as optimal anchor point (6504K, CIE XYZ [0.95047, 1.0, 1.08883])
- Adobe DNG system provides license-free camera profiles for 350+ models
- 2024 developments: AI-powered white balance in 100M+ devices, learnable CCMs
- CIEDE2000 (ΔE00) established as most perceptually meaningful color difference metric

📋 **PRD Enhancements:**
- Added comprehensive API design with ColorResult struct
- Specified D65 anchor point and calibration constants
- Detailed project structure with modular architecture
- Enhanced dependency stack with research-backed choices
- Performance targets: 100ms analysis, ΔE < 3.0 accuracy
- Added flash handling strategy (deferred) and smartphone context

🏗️ **Project Setup:**
- Git repository initialized
- Rust crate structure created with cargo
- Task-master system initialized (manual fallback due to AI service issue)
- Template files copied and customized for project

**Next**: Create directory structure and setup Cargo.toml dependencies