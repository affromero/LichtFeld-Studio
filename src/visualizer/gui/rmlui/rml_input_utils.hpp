/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/Input.h>

namespace lfs::vis::gui {

    Rml::Input::KeyIdentifier imguiKeyToRml(int key);
    int buildRmlModifiers();

} // namespace lfs::vis::gui
