from nicegui import ui

def apply_theme():
    ui.add_css('''
    * {
        font-size: 18px !important;
    }
    body, .q-page, .nicegui-content {
        font-size: 18px !important;
    }
    .q-field__label, .q-item__label, .q-btn, .q-tab, input, textarea {
        font-size: 18px !important;
    }
    /* Scale checkboxes to match text size */
    .q-checkbox__svg {
        width: 32px !important;
        height: 32px !important;
    }
    .q-checkbox__inner {
        font-size: 32px !important;
        width: 32px !important;
        height: 32px !important;
    }
    ''')