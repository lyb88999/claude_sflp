# Iridium_TLE_Generator.py
import math

def calculate_checksum(line):
    return str((sum(1 if c == '-' else int(c) if c.isdigit() else 0 
                for c in line[:68]) % 10))

def generate_iridium():
    tle_data = []
    sat_per_plane = 11
    planes = 6
    base_norad = 40001
    ecc = 0.001  # 标准低地球轨道偏心率
    
    for plane in range(1, planes+1):
        raan = 60 * (plane-1)  # 升交点赤经
        for sat in range(1, sat_per_plane+1):
            norad = base_norad + (plane-1)*11 + (sat-1)
            mean_anomaly = 360 * (sat-1)/sat_per_plane
            
            # 生成国际标识符
            intl_desig = f"24{plane:02d}{chr(64+sat)}"
            
            # 构建TLE
            line1 = f"1 {norad:05d}U {intl_desig}  24001.00000000  .00000000  00000-0  00000-0 0  999"
            line2 = f"2 {norad:05d} 86.4000 {raan:8.4f} {ecc:.4f} 0000000   0.0000 {mean_anomaly:8.4f} 14.34920000"
            
            # 计算校验和
            line1 += calculate_checksum(line1)
            line2 += calculate_checksum(line2)
            
            tle_data.append(f"Iridium {plane}-{sat}")
            tle_data.append(line1)
            tle_data.append(line2)
    
    return '\n'.join(tle_data)

# 生成并保存文件
with open("Iridium_TLEs.txt", "w") as f:
    f.write(generate_iridium())

print("TLE生成完成，文件已保存为Iridium_TLEs.txt")