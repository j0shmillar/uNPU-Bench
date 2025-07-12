/*
 * Copyright 2020-2022 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "timer.h"
#include "fsl_gpio.h"
#include "fsl_clock.h"
#include "fsl_cmc.h" 
#include "fsl_spc.h" 
#include "fsl_vbat.h"

#define APP_CMC           CMC0
#define APP_RAM_ARRAYS_DS (0x3F0077FE)
#define APP_RAM_ARRAYS_PD (0x3F0077FE)

#define APP_LPTMR             LPTMR0
#define APP_LPTMR_IRQN        LPTMR0_IRQn
#define APP_LPTMR_IRQ_HANDLER LPTMR0_IRQHandler
#define APP_LPTMR_CLK_SOURCE  (16000UL)

#define APP_SPC                           SPC0
#define APP_SPC_LPTMR_LPISO_VALUE         (0x1EU) /* VDD_USB, VDD_P2, VDD_P3, VDD_P4. */
#define APP_SPC_LPTMR_ISO_DOMAINS         "VDD_USB, VDD_P2, VDD_P3, VDD_P4"
#define APP_SPC_WAKEUP_BUTTON_LPISO_VALUE (0x3EU) /* VDD_USB, VDD_P2, VDD_P3, VDD_P4, VBAT. */
#define APP_SPC_WAKEUP_BUTTON_ISO_DOMAINS "VDD_USB, VDD_P2, VDD_P3, VDD_P4, VBAT."
#define APP_SPC_MAIN_POWER_DOMAIN         (kSPC_PowerDomain0)
#define APP_SPC_WAKE_POWER_DOMAIN         (kSPC_PowerDomain1)

#define APP_VBAT             VBAT0
#define APP_VBAT_IRQN        VBAT0_IRQn
#define APP_VBAT_IRQ_HANDLER VBAT0_IRQHandler

static void APP_SetSPCConfiguration(void)
{
    status_t status;

    spc_active_mode_regulators_config_t activeModeRegulatorOption;

#if defined(APP_INVALIDATE_CACHE)
    APP_INVALIDATE_CACHE;
#endif /* defined(APP_INVALIDATE_CACHE) */

    /* Disable all modules that controlled by SPC in active mode. */
    SPC_DisableActiveModeAnalogModules(APP_SPC, kSPC_controlAllModules);

    /* Disable LVDs and HVDs */
    SPC_EnableActiveModeCoreHighVoltageDetect(APP_SPC, false);
    SPC_EnableActiveModeCoreLowVoltageDetect(APP_SPC, false);
    SPC_EnableActiveModeSystemHighVoltageDetect(APP_SPC, false);
    SPC_EnableActiveModeSystemLowVoltageDetect(APP_SPC, false);
    SPC_EnableActiveModeIOHighVoltageDetect(APP_SPC, false);
    SPC_EnableActiveModeIOLowVoltageDetect(APP_SPC, false);

    while(SPC_GetBusyStatusFlag(APP_SPC))
        ;

    activeModeRegulatorOption.bandgapMode = kSPC_BandgapEnabledBufferDisabled;
    activeModeRegulatorOption.lpBuff      = false;
    /* DCDC output voltage is 1.1V in active mode. */
    activeModeRegulatorOption.DCDCOption.DCDCVoltage           = kSPC_DCDC_NormalVoltage;
    activeModeRegulatorOption.DCDCOption.DCDCDriveStrength     = kSPC_DCDC_NormalDriveStrength;
    activeModeRegulatorOption.SysLDOOption.SysLDOVoltage       = kSPC_SysLDO_NormalVoltage;
    activeModeRegulatorOption.SysLDOOption.SysLDODriveStrength = kSPC_SysLDO_LowDriveStrength;
    activeModeRegulatorOption.CoreLDOOption.CoreLDOVoltage     = kSPC_CoreLDO_NormalVoltage;
#if defined(FSL_FEATURE_SPC_HAS_CORELDO_VDD_DS) && FSL_FEATURE_SPC_HAS_CORELDO_VDD_DS
    activeModeRegulatorOption.CoreLDOOption.CoreLDODriveStrength = kSPC_CoreLDO_NormalDriveStrength;
#endif /* FSL_FEATURE_SPC_HAS_CORELDO_VDD_DS */

    status = SPC_SetActiveModeRegulatorsConfig(APP_SPC, &activeModeRegulatorOption);
    /* Disable Vdd Core Glitch detector in active mode. */
    SPC_DisableActiveModeVddCoreGlitchDetect(APP_SPC, true);
    if (status != kStatus_Success)
    {
        return;
    }
    while (SPC_GetBusyStatusFlag(APP_SPC))
        ;

    SPC_DisableLowPowerModeAnalogModules(APP_SPC, kSPC_controlAllModules);
    SPC_SetLowPowerWakeUpDelay(APP_SPC, 0xFF);
    spc_lowpower_mode_regulators_config_t lowPowerRegulatorOption;

    lowPowerRegulatorOption.lpIREF      = false;
    lowPowerRegulatorOption.bandgapMode = kSPC_BandgapDisabled;
    lowPowerRegulatorOption.lpBuff      = false;
    /* Enable Core IVS, which is only useful in power down mode. */
    lowPowerRegulatorOption.CoreIVS = true;
    /* DCDC output voltage is 1.0V in some low power mode(Deep sleep, Power Down). DCDC is disabled in Deep Power Down.
     */
    lowPowerRegulatorOption.DCDCOption.DCDCVoltage             = kSPC_DCDC_MidVoltage;
    lowPowerRegulatorOption.DCDCOption.DCDCDriveStrength       = kSPC_DCDC_LowDriveStrength;
    lowPowerRegulatorOption.SysLDOOption.SysLDODriveStrength   = kSPC_SysLDO_LowDriveStrength;
    lowPowerRegulatorOption.CoreLDOOption.CoreLDOVoltage       = kSPC_CoreLDO_MidDriveVoltage;
    lowPowerRegulatorOption.CoreLDOOption.CoreLDODriveStrength = kSPC_CoreLDO_LowDriveStrength;

    status = SPC_SetLowPowerModeRegulatorsConfig(APP_SPC, &lowPowerRegulatorOption);
    /* Disable Vdd Core Glitch detector in low power mode. */
    SPC_DisableLowPowerModeVddCoreGlitchDetect(APP_SPC, true);
    if (status != kStatus_Success)
    {
        return;
    }
    while (SPC_GetBusyStatusFlag(APP_SPC))
        ;

    /* Disable LDO_CORE since it is bypassed. */
    SPC_EnableCoreLDORegulator(APP_SPC, false);

    /* Enable Low power request output to observe the entry/exit of
     * low power modes(including: deep sleep mode, power down mode, and deep power down mode).
     */
    spc_lowpower_request_config_t lpReqConfig = {
        .enable   = true,
        .polarity = kSPC_LowTruePolarity,
        .override = kSPC_LowPowerRequestNotForced,
    };

    SPC_SetLowPowerRequestConfig(APP_SPC, &lpReqConfig);
}

static void APP_SetVBATConfiguration(void)
{
    if (VBAT_CheckFRO16kEnabled(APP_VBAT) == false)
    {
        /* In case of FRO16K is not enabled, enable it firstly. */
        VBAT_EnableFRO16k(APP_VBAT, true);
    }
    VBAT_UngateFRO16k(APP_VBAT, kVBAT_EnableClockToVddSys);

    /* Disable Bandgap to save current consumption. */
    if (VBAT_CheckBandgapEnabled(APP_VBAT))
    {
        VBAT_EnableBandgapRefreshMode(APP_VBAT, false);
        VBAT_EnableBandgap(APP_VBAT, false);
    }
}

static void APP_SetCMCConfiguration(void)
{
    /* Disable low power debug. */
    CMC_EnableDebugOperation(APP_CMC, false);
    /* Allow all power mode */
    CMC_SetPowerModeProtection(APP_CMC, kCMC_AllowAllLowPowerModes);

#if (defined(FSL_FEATURE_MCX_CMC_HAS_NO_FLASHCR_WAKE) && FSL_FEATURE_MCX_CMC_HAS_NO_FLASHCR_WAKE)
    /* From workaround of errata 051993, enable doze feature. */
    CMC_ConfigFlashMode(APP_CMC, true, false);
#else
    /* Disable flash memory accesses and place flash memory in low-power state whenever the core clock
       is gated. And an attempt to access the flash memory will cause the flash memory to exit low-power
       state for the duration of the flash memory access. */
    CMC_ConfigFlashMode(APP_CMC, true, true, false);
#endif
}

//Lower the active current. Code comes from the power_mode_switch_II SDK example
void lower_active_power()
{

	APP_SetVBATConfiguration();
    APP_SetSPCConfiguration();
    APP_SetCMCConfiguration();

    /* Disable unused clocks */
    CLOCK_DisableClock(kCLOCK_Rom);
    CLOCK_DisableClock(kCLOCK_PkcRam);
    CLOCK_DisableClock(kCLOCK_Gdet);
    CLOCK_DisableClock(kCLOCK_Pkc);
    CLOCK_DisableClock(kCLOCK_Css);
}